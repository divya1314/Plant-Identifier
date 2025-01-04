import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
dataset_path = r'D:\TENSOR EXPO\combined_dataset'  # Path to your combined dataset
train_path = r'D:\TENSOR EXPO\split_dataset\train'
test_path = r'D:\TENSOR EXPO\split_dataset\test'

# Create directories for train/test splits
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Iterate through each class
for class_name in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_name)
    if not os.path.isdir(class_path):
        continue

    # Get all file names for the class
    files = os.listdir(class_path)
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)

    # Create class directories in train/test folders
    os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_path, class_name), exist_ok=True)

    # Move files to respective folders
    for file in train_files:
        shutil.copy(os.path.join(class_path, file), os.path.join(train_path, class_name, file))
    for file in test_files:
        shutil.copy(os.path.join(class_path, file), os.path.join(test_path, class_name, file))

print("Dataset split completed!")
