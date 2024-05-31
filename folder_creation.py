import os

# Define the folder structure
folders = [
    "result",
    "result/truthfulqa",
    "result/truthfulqa/ig",
    "result/truthfulqa/magnitude",
    "result/truthfulqa/res",
    "result/truthfulqa/res/both"
]

# Create the folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created folder: {folder}")

print("All folders created successfully.")
