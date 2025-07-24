import ml_collections
import os


def get_enhancer_config(
    parent_dir,
    state,
    which_model,
    cls_free_guidance=False,
    train_fbd=False,
    distill_data_path=None,
    sampler_name="euler",
    do_purity_sampling=False,
    dt=0.01,
    use_tag=True,
    discrete_time=False,
    num_timesteps=100,
):
    """
    Config for training and evaluating on enhancer dataset
    Can create denoising model, noisy classifier or clean classifier

    Args:
        train_fbd: Whether to compute fbd at train time
    """
    # Parse the arguments to specify which config file to create
    assert state in ["train", "eval"]
    assert which_model in ["denoising", "cls_noisy", "cls_clean"]
    # Predictor-free guidance or FBD evaluation are irrelavant for training classifier
    if which_model in ["cls_noisy", "cls_clean"]:
        assert not train_fbd
        assert not cls_free_guidance

    # General config
    config = ml_collections.ConfigDict()

    # The directory for saving the run directories
    # for both training and sampling
    config.parent_dir = parent_dir
    save_dir = os.path.join(parent_dir, "outputs/")

    config.discrete_time = discrete_time
    # Number of timesteps is only relevant for discrete time models
    config.num_timesteps = num_timesteps

    # Name experiment
    if state == "train":
        # Folder name for traininig
        experiment_name = f"enhancer-train-{which_model}"
        if cls_free_guidance:
            experiment_name += "-pfg"
    else:
        # Folder name for sampling
        experiment_name = f"enhancer-sample-{sampler_name}"
        if cls_free_guidance:
            experiment_name += "-pfg"
        else:
            experiment_name += "-pg"
            if not use_tag:
                experiment_name += "_exact"
    if discrete_time:
        experiment_name += f"-d3pm_{config.num_timesteps}"

    # Train on augmented dataset only relevant for noisy classifier
    if distill_data_path is not None:
        config.distill_data_path = distill_data_path
        assert which_model == "cls_noisy"

    config.experiment_name = experiment_name

    config.save_location = save_dir
    config.state = state
    config.init_model_path = None
    config.device = "cuda"
    config.distributed = False
    config.num_gpus = 1
    config.eps_ratio = 1e-9
    config.seed = 43

    # Configs for data
    config.data = data = ml_collections.ConfigDict()
    if which_model == "cls_clean":
        data.S = 4
    else:
        data.S = 5  # Alphabet size (including mask)
    data.shape = 500  # Length of the sequence
    data.mel_enhancer = False  # If using the Melanoma dataset, else use fly brain
    data.num_classes = 81
    data.categorical = True
    # Whether the data is noised
    # This is False except for the unnoised classifier ('cls_clean')
    # We shouldn't need to since we can use the pretrained ckpt
    data.clean_data = which_model == "cls_clean"

    # Define model, either the denoising model or the property model
    # If property model, then model.classifier = True
    config.model = model = ml_collections.ConfigDict()
    model.name = "cnn"
    model.classifier = which_model != "denoising"
    # Settings for CFG
    model.cls_free_guidance = cls_free_guidance
    model.cls_free_noclass_ratio = 0.3
    model.hidden_dim = 128
    model.p_dropout = 0.0
    if which_model == "denoising":
        model.num_cnn_stacks = 4
    else:
        model.num_cnn_stacks = 1

    if state == "train":
        # Training specific configs

        # Configs for training setup
        config.training = training = ml_collections.ConfigDict()
        if which_model == "denoising":
            training.batch_size = 256
            training.n_iters = 300000
            training.warmup = 500
            training.lr = 5e-4
        elif which_model == "cls_noisy":
            training.batch_size = 256
            training.n_iters = 300000
            training.warmup = 0
            training.lr = 1e-3
        elif which_model == "cls_clean":
            training.batch_size = 128
            training.n_iters = 10480
            training.warmup = 0
            training.lr = 1e-3
        training.clip_grad = True

        if train_fbd:
            # Evaluate and save checkpoint for fbd
            # The pretrained clean classifier (train with no noise) used for evaluation
            config.cls_clean_model_train_config_path = os.path.join(
                parent_dir,
                "workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/lightning_logs/version_0/hparams.yaml",
            )
            config.cls_clean_model_checkpoint_path = os.path.join(
                parent_dir,
                "workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/epoch=15-step=10480.ckpt",
            )

        # Configs for logging
        config.saving = saving = ml_collections.ConfigDict()
        saving.enable_preemption_recovery = False
        saving.preemption_start_day_YYYYhyphenMMhyphenDD = None
        saving.checkpoint_freq = 1000
        saving.num_checkpoints_to_keep = 2
        saving.checkpoint_archive_freq = 20000
        saving.prepare_to_resume_after_timeout = False

    else:
        # Evaluation / sampling specific configs
        config.eval_name = "enhancer"
        # The directory where pretrained model weights are stored
        model_weights_dir = os.path.join(parent_dir, "model_weights")

        # The pretrained clean classifier used for evaluation, provided by Dirichlet FM
        # This we never change
        config.cls_clean_model_train_config_path = os.path.join(
            parent_dir,
            "workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/lightning_logs/version_0/hparams.yaml",
        )
        config.cls_clean_model_checkpoint_path = os.path.join(
            parent_dir,
            "workdir/clsDNAclean_cnn_1stack_2023-12-30_15-01-30/epoch=15-step=10480.ckpt",
        )

        if cls_free_guidance:
            config.denoising_model_checkpoint_path = os.path.join(
                model_weights_dir, "pfg/model_ckpt.pt"
            )
            config.denoising_model_train_config_path = os.path.join(
                model_weights_dir, "pfg/config.yaml"
            )
            config.guide_temps = [1.0, 0.5, 0.2, 0.1, 0.05]
        else:
            if discrete_time:
                ## D3PM models
                config.denoising_model_checkpoint_path = os.path.join(
                    model_weights_dir, "digress/denoising_model_ckpt.pt"
                )
                config.denoising_model_train_config_path = os.path.join(
                    model_weights_dir, "digress/denoising_model_config.yaml"
                )
                ## Noisy classifier
                config.cls_model_checkpoint_path = os.path.join(
                    model_weights_dir, "digress/noisy_classifier_ckpt.pt"
                )
                config.cls_model_train_config_path = os.path.join(
                    model_weights_dir, "digress/noisy_classifier_config.yaml"
                )
            else:
                ### Flow matching models
                ## Denoising model
                config.denoising_model_checkpoint_path = os.path.join(
                    model_weights_dir, "pg/denoising_model_ckpt.pt"
                )
                config.denoising_model_train_config_path = os.path.join(
                    model_weights_dir, "pg/denoising_model_config.yaml"
                )
                ## Noisy classifier
                config.cls_model_checkpoint_path = os.path.join(
                    model_weights_dir, "pg/noisy_classifier_ckpt.pt"
                )
                config.cls_model_train_config_path = os.path.join(
                    model_weights_dir, "pg/noisy_classifier_config.yaml"
                )

            ## Guidance temperature
            config.guide_temps = [1.0, 0.5, 0.2, 0.1, 0.05]

        config.target_classes = [16, 5, 4, 2, 33, 68, 9, 12]
        config.sampler = sampler = ml_collections.ConfigDict()
        sampler.name = sampler_name
        sampler.batch_size = 500
        sampler.dt = dt
        sampler.noise = 0.0
        sampler.x1_temp = 1.0
        sampler.purity_temp = 1.0
        sampler.do_purity_sampling = do_purity_sampling
        sampler.argmax_final = True
        sampler.max_t = 1
        sampler.use_tag = use_tag

    return config
