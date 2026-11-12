
from huggingface_hub import snapshot_download

repo_id = "swan07/authorship-verification"

# Save everything inside your existing folder
local_dir = r"D:\unsupervised learnnig\DarkID_Dataset"

print("Downloading the authorship verification dataset...")

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir
)

print("\nDownload complete!")
print(f"Dataset location: {local_dir}")