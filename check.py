import os

folder = r"D:\College Work\UnsupervisedProject"

print("Project folder contents:")
print(os.listdir(folder))

dataset_folder = os.path.join(folder, "DarkID_Dataset")

print("\nDataset folder exists:", os.path.exists(dataset_folder))

if os.path.exists(dataset_folder):
    print("\nDataset folder contents:")
    for item in os.listdir(dataset_folder):
        print(item)
else:
    print("\nDarkID_Dataset was not found at the expected location.")