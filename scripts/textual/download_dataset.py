import os
from datasets import load_dataset

os.makedirs('data', exist_ok=True)

# Load the subset
dataset = load_dataset("cambridgeltl/DARE", "1_correct")

# Get validation split
val_split = dataset["validation"]

# Drop the heavy image column
val_split_noimg = val_split.remove_columns(["img"])

# Save to CSV
val_split_noimg.to_csv("data/[without images]1_correct_validation.csv")

print("Saved: data/[without images]1_correct_validation.csv")