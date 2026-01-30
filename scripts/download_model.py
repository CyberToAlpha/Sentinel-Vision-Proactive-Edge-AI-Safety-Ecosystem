from huggingface_hub import hf_hub_download
import shutil
import os

repo_id = "keremberke/yolov8s-protective-equipment-detection"
filename = "best.pt"

print(f"Downloading {filename} from {repo_id}...")
try:
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    # Copy to local directory as safety_model.pt
    target_path = "safety_model.pt"
    shutil.copy(model_path, target_path)
    print(f"Model saved to {target_path}")
except Exception as e:
    print(f"Failed to download model: {e}")
    # Try finding 'yolov8s.pt' if best.pt defaults fail, though best.pt is standard for these repos

