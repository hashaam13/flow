from selene_utils import MemmapGenome
#creates .mmap file. Only run once
genome = MemmapGenome(
    input_path="/home/hmuhammad/flow/data/promoter_design/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
    memmapfile="/home/hmuhammad/flow/data/promoter_design/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap",
    blacklist_regions="hg38"
)
genome._unpicklable_init()