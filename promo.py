from utils.promoter_dataset import PromoterDataset   # assuming you saved that class into promoter_dataset.py

# Example: build training dataset
train_data = PromoterDataset(split="train", n_tsses=100000, rand_offset=10)

# Inspect one sample
x = train_data[0]   # numpy array: (seq_length, 6)  (4 bases + 2 CAGE channels)
print(x.shape)
print(len(train_data))
