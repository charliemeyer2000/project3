"""Upload training dataset to Modal volume (run once)."""

import modal

app = modal.App("upload-project3-dataset")
volume = modal.Volume.from_name("project3-kd-data", create_if_missing=True)

@app.function(volumes={"/data": volume}, timeout=1800)
def upload():
    """Upload dataset to Modal volume."""
    import os
    import shutil
    from pathlib import Path
    
    print("Checking if dataset exists in volume...")
    if os.path.exists("/data/training_dataset"):
        print("✓ Dataset already exists in volume")
        # Count images
        count = sum(1 for root, dirs, files in os.walk("/data/training_dataset") 
                   for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        print(f"  Found {count} images")
        return
    
    print("Dataset not found. Please upload manually:")
    print("  modal volume put project3-kd-data training_dataset /data/training_dataset")
    
    volume.commit()

@app.local_entrypoint()
def main():
    """Check dataset in volume."""
    upload.remote()
    
    print("\nTo upload dataset manually, run:")
    print("  modal volume put project3-kd-data training_dataset /data/training_dataset")

