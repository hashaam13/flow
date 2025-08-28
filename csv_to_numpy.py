import pandas as pd
import numpy as np

# Define the character to integer mapping
char_to_int = {"A": 0, "C": 1, "G": 2, "T": 3}

# Read the CSV file
df = pd.read_csv('data/dirichlet_data/val_0.csv')  # Replace with your CSV file path

# Function to convert sequence string to integer array
def sequence_to_int(seq):
    return np.array([char_to_int[char] for char in seq], dtype=np.int64)

# Convert all val_seq entries to integer arrays
sequences_array = np.array([sequence_to_int(seq) for seq in df['val_seq']])

# Save the numpy array
np.save('output_samples/dirichlet_0_cfg3.npy', sequences_array)

print(f"Converted {len(sequences_array)} sequences")
print(f"Array shape: {sequences_array.shape}")
print(f"Array dtype: {sequences_array.dtype}")
