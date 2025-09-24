import torch
import torch.nn.functional as F
from torch import Tensor

from flow_matching.path.path import ProbPath
from flow_matching.path.path_sample import DiscretePathSample
from flow_matching.path.scheduler import ConvexScheduler
from flow_matching.utils import expand_tensor_like, unsqueeze_to_match

# -------------------------
# Optional advanced helper: tiny Laplacian-based KO flux->rates (for future use)
# -------------------------
def ko_rates_from_p_and_dpdt(p: Tensor, dpdt: Tensor, w_ab: Tensor = None, reg: float = 1e-6) -> Tensor:
    """
    Compute a rate (generator) matrix v from p and dpdt using a small Laplacian solve.
    p: (..., K) probability vector (per-position)
    dpdt: (..., K) time derivative of p
    returns v: (..., K, K) rate matrix (rows sum to 0)
    NOTE: This is an advanced utility (slow if used naively) and not used by default.
    """
    # Flatten batch dims for solve convenience
    orig_shape = p.shape
    *batch, K = p.shape
    Bflat = int(torch.tensor(batch).prod().item()) if len(batch) > 0 else 1
    p_flat = p.reshape(-1, K)
    dpdt_flat = dpdt.reshape(-1, K)

    device = p.device
    if w_ab is None:
        w = torch.ones(K, K, device=device) - torch.eye(K, device=device)
    else:
        w = w_ab.to(device)
    L = torch.diag(w.sum(dim=1)) - w  # (K,K)

    L_reg = L + reg * torch.eye(K, device=device)
    # Solve for phi: L_reg @ phi = dpdt  (we do for each row separately)
    # For small K (DNA K=4) this is cheap
    phi = torch.linalg.solve(L_reg, dpdt_flat.T).T  # (Bflat, K)

    diff = phi.unsqueeze(1) - phi.unsqueeze(2)  # (Bflat, K, K) -> phi[b,i] - phi[b,j]
    p_expand = p_flat.unsqueeze(2)  # (Bflat, K, 1)
    Phi = p_expand * w.unsqueeze(0) * diff  # (Bflat, K, K)
    Phi = F.relu(Phi)

    v = torch.zeros_like(Phi)
    denom = p_flat.clamp(min=1e-12).unsqueeze(1)  # (Bflat,1,K) -> careful broadcasting
    # v[a,b] = Phi[a,b] / p[a]
    v = Phi / (p_flat.unsqueeze(2).clamp(min=1e-12))
    v = v.clone()
    # set diagonal
    v = v - torch.diag_embed(v.sum(dim=-1))
    v = v.reshape(*batch, K, K)
    return v


# -------------------------
# KineticOptimalProbPath
# -------------------------
class KineticOptimalProbPath(ProbPath):
    r"""A practical, pluggable kinetic-optimal–inspired discrete probability path.

    This class aims to be API-compatible with the repo's ProbPath interface and
    with MixtureDiscreteProbPath, so you can drop it in as a replacement. It
    produces per-position categorical marginals:

        p_{t,i} = (1 - s_i(t)) * one_hot(x1^i) + s_i(t) * pi_i

    where:
      - s_i(t) = alpha(t) * scale_i
      - pi_i is a per-position prior (L, K) or uniform if None
      - scale_i is a deterministic function of pi_i (dataset-dependent bias),
        normalized to [0.2, 1.0] to avoid freezing.

    This class implements:
      - sample(x_0, x_1, t) -> DiscretePathSample (returns sampled x_t)
      - posterior_to_velocity(posterior_logits, x_t, t) -> velocity tensor
        (approximate, compatible with existing MixturePathGeneralizedKL loss)

    Args:
        scheduler (ConvexScheduler): same scheduler interface used elsewhere.
        position_priors (Tensor or None): optional tensor (L, K) of per-position priors.
                                            dtype=float, sums to 1 on last dim.
        min_scale, max_scale: bounds for the per-position scale factor.
    """

    def __init__(self, scheduler: ConvexScheduler, position_priors: Tensor = None, min_scale: float = 0.2, max_scale: float = 1.0):
        assert isinstance(scheduler, ConvexScheduler), "scheduler must be a ConvexScheduler"
        self.scheduler = scheduler
        # position_priors expected shape (L, K) or None
        if position_priors is not None:
            assert position_priors.dim() == 2, "position_priors must be shape (L, K)"
            self.position_priors = position_priors.clone().float()
        else:
            self.position_priors = None
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

    # -------------------------
    # internal helpers
    # -------------------------
    def _infer_vocab_size(self, x_0: Tensor, x_1: Tensor) -> int:
        return int(max(int(x_0.max().item()), int(x_1.max().item()))) + 1

    def _get_pi_for_batch(self, x_1: Tensor) -> Tensor:
        """
        Return pi expanded to (B, L, K). If no priors provided, return uniform.
        """
        B, L = x_1.shape
        K = self._infer_vocab_size(x_0=x_1, x_1=x_1)  # safe fallback
        if self.position_priors is None:
            pi = torch.full((L, K), 1.0 / K, device=x_1.device, dtype=torch.float32)
        else:
            # if provided K mismatches, try to adapt or raise
            if self.position_priors.shape[1] != K:
                # try to expand/crop if possible
                if self.position_priors.shape[1] < K:
                    # pad uniform for missing classes
                    pad = torch.full((L, K - self.position_priors.shape[1]), 1.0 / K, device=self.position_priors.device)
                    pi = torch.cat([self.position_priors.to(pad.device), pad], dim=1)
                else:
                    pi = self.position_priors[:, :K].to(x_1.device)
            else:
                pi = self.position_priors.to(x_1.device)
        # expand to batch
        return pi.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, L, K)

    def _one_hot(self, x: Tensor, K: int) -> Tensor:
        return F.one_hot(x.long(), num_classes=K).float()

    def _compute_scale_from_pi(self, pi: Tensor) -> Tensor:
        """
        Compute per-position scale in [min_scale, max_scale] based on pi.
        We use entropy(pi) as a heuristic: high entropy -> less confident prior -> larger scale.
        pi shape: (B, L, K) or (L,K) -> we support (B,L,K).
        returns scale (B,L)
        """
        if pi.dim() == 2:
            pi_ = pi.unsqueeze(0)  # (1,L,K)
        else:
            pi_ = pi  # (B,L,K)
        # entropy per position
        eps = 1e-12
        ent = - (pi_ * (pi_.clamp(min=eps)).log()).sum(dim=-1)  # (B,L)
        # normalize ent to [0,1] per batch (avoid degenerate)
        ent_min, ent_max = ent.min(dim=1, keepdim=True)[0], ent.max(dim=1, keepdim=True)[0]
        denom = (ent_max - ent_min).clamp(min=1e-6)
        ent_norm = (ent - ent_min) / denom  # (B,L)
        # invert mapping so that low entropy -> scale closer to min_scale, high entropy -> closer to max_scale
        scale = self.min_scale + (self.max_scale - self.min_scale) * ent_norm
        return scale  # (B,L)

    # -------------------------
    # core API: sample
    # -------------------------
    def sample(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> DiscretePathSample:
        """
        x_0: (B, L) integer token tensor (source)
        x_1: (B, L) integer token tensor (target)
        t:   (B,) times in [0,1]
        returns DiscretePathSample(x_t, x_1, x_0, t)
        """
        self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)

        B, L = x_1.shape
        K = self._infer_vocab_size(x_0=x_0, x_1=x_1)

        # one-hot of target
        onehot_x1 = self._one_hot(x_1, K=K)  # (B,L,K)
        pi = self._get_pi_for_batch(x_1)     # (B,L,K)

        # scheduler alpha and derivative
        sched_out = self.scheduler(t)  # object with attributes alpha_t, d_alpha_t (ConvexScheduler API)
        alpha_t = sched_out.alpha_t    # (B,)
        # expand to (B, L)
        alpha_exp = expand_tensor_like(input_tensor=alpha_t, expand_to=x_1)  # uses repo helper

        # compute scale per position (depends only on pi)
        scale = self._compute_scale_from_pi(pi)  # (B,L)

        # local s_i(t)
        s_local = (alpha_exp * scale).clamp(0.0, 1.0)  # (B,L)

        # p_t per-position mixture
        s_local_unsq = s_local.unsqueeze(-1)  # (B,L,1)
        p_t = (1.0 - s_local_unsq) * onehot_x1 + s_local_unsq * pi  # (B,L,K)
        p_t = p_t.clamp(min=1e-12)
        p_t = p_t / p_t.sum(dim=-1, keepdim=True)

        # sample categorical per-position
        probs = p_t.view(B * L, K)
        # numerically stable normalization
        probs = probs / probs.sum(dim=-1, keepdim=True)
        samples = torch.multinomial(probs, 1).view(B, L)  # (B,L) ints

        return DiscretePathSample(x_t=samples.to(x_1.device), x_1=x_1, x_0=x_0, t=t)

    # -------------------------
    # core API: posterior_to_velocity (approximate, API-compatible)
    # -------------------------
    def posterior_to_velocity(self, posterior_logits: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        r"""Convert the factorized posterior to velocity (approximate, API-compatible).

        posterior_logits: (..., K) logits of p(X_1 | X_t)
        x_t: integer tokens at time t, shape (...), same leading dims as posterior_logits minus last dim
        t: times (B,)

        Returns:
            Tensor of same shape as posterior_logits representing velocity u_t.
        """
        posterior = torch.softmax(posterior_logits, dim=-1)  # (..., K)
        vocabulary_size = posterior.shape[-1]

        # x_t one-hot
        x_t_oh = F.one_hot(x_t.long(), num_classes=vocabulary_size).float()  # (..., K)

        # scheduler alpha and derivative
        t_expanded = t
        # unsqueeze_to_match expects t and target shape, as used in repo mixture
        t_match = unsqueeze_to_match(source=t_expanded, target=x_t_oh)
        sched_out = self.scheduler(t_match)  # should broadcast
        alpha_t = sched_out.alpha_t
        d_alpha_t = sched_out.d_alpha_t

        # get pi and scale at the right broadcasted shape
        # need to infer batch/length dims from x_t
        # x_t shape can be (B,L); x_t_oh is (B,L,K)
        # build pi of shape (B,L,K)
        # for convenience we reconstruct a dummy x_1 from x_t (not needed), use _get_pi_for_batch by creating a batch of zeros of same shape
        # but _get_pi_for_batch needs x_1 to get B and L; so we create a dummy tensor
        if x_t.dim() == 1:
            # if x_t is (B,), assume single position -> reshape
            raise ValueError("posterior_to_velocity expects x_t with at least 2 dims (B,L).")

        B, L = x_t.shape
        dummy_x1 = x_t  # only for shape inference
        pi = self._get_pi_for_batch(dummy_x1)  # (B,L,K)
        scale = self._compute_scale_from_pi(pi)  # (B,L)

        # expand alpha and d_alpha to (B,L)
        alpha_expanded = expand_tensor_like(input_tensor=alpha_t, expand_to=x_t)
        d_alpha_expanded = expand_tensor_like(input_tensor=d_alpha_t, expand_to=x_t)

        # local s_i and dsdt
        s_local = (alpha_expanded * scale).clamp(0.0, 1.0)  # (B,L)
        dsdt_local = d_alpha_expanded * scale  # (B,L)

        # unsqueeze to match last dim
        s_local_u = s_local.unsqueeze(-1)    # (B,L,1)
        dsdt_local_u = dsdt_local.unsqueeze(-1)  # (B,L,1)

        # velocity: approximate formula analogous to mixture path:
        # u_t = (dsdt / (1 - s_local)) * (posterior - x_t_onehot)
        denom = (1.0 - s_local_u).clamp(min=1e-6)
        coeff = dsdt_local_u / denom  # (B,L,1)
        # posterior and x_t_oh must have shape (B,L,K) matching coeff -- ensure broadcasting:
        # posterior may have shape (..., K) where ... corresponds to (B,L)
        u_t = coeff * (posterior - x_t_oh)

        return u_t
