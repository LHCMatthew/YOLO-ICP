import random, glob
import shutil
from pathlib import Path

base_path = "../../../train_pbr"

# Loop through all 50 folders (000000 to 000049)
for folder_idx in range(50):
    folder_name = f"{folder_idx:06d}"
    path = f"{base_path}/{folder_name}"
    
    print(f"\n{'='*60}")
    print(f"Processing folder: {folder_name}")
    print(f"{'='*60}")
    
    # Check if rgb folder exists
    rgb_path = Path(f"{path}/rgb")
    if not rgb_path.exists():
        print(f"Warning: {path}/rgb does not exist, skipping this folder")
        continue
    
    # Rename rgb folder to images
    images_path = Path(f"{path}/images")
    if not images_path.exists():
        rgb_path.rename(images_path)
        print(f"Renamed rgb folder to images")
    else:
        print(f"images folder already exists, using existing folder")
    
    # Get images from the images folder
    imgs = sorted(glob.glob(f"{path}/images/*.jpg"))
    if len(imgs) == 0:
        print(f"No images found in {path}/images, skipping this folder")
        continue
    
    random.shuffle(imgs)
    n = len(imgs)
    train = imgs[:int(0.7*n)]
    val   = imgs[int(0.7*n):int(0.9*n)]
    test  = imgs[int(0.9*n):]
    
    print(f"Found {n} images: {len(train)} train, {len(val)} val, {len(test)} test")
    
    # Create folders and move images
    for name, lst in zip(["train","val","test"], [train,val,test]):
        # Create folder if it doesn't exist
        folder = Path(f"{path}/images/{name}")
        folder.mkdir(parents=True, exist_ok=True)
        
        # Move images to the respective folder
        for img_path in lst:
            img = Path(img_path)
            dest = folder / img.name
            shutil.move(img_path, dest)
            print(f"Moved {img.name} to {name} folder")
    
    # Create labels train/val/test folders
    labels_path = Path(f"{path}/labels")
    if not labels_path.exists():
        print(f"Warning: {path}/labels does not exist, skipping label organization for this folder")
        continue
    
    (labels_path / "train").mkdir(parents=True, exist_ok=True)
    (labels_path / "val").mkdir(parents=True, exist_ok=True)
    (labels_path / "test").mkdir(parents=True, exist_ok=True)
    
    # Organize labels according to their corresponding images
    train_imgs = sorted(glob.glob(f"{path}/images/train/*.jpg"))
    val_imgs   = sorted(glob.glob(f"{path}/images/val/*.jpg"))
    test_imgs  = sorted(glob.glob(f"{path}/images/test/*.jpg"))
    label_imgs = sorted(glob.glob(f"{path}/labels/*.txt"))
    
    # Check if the labels correspond to val, test, or train images, move to appropriate folder
    for label in label_imgs:
        label_stem = Path(label).stem
        if label_stem in [Path(p).stem for p in train_imgs]:
            shutil.move(label, f"{path}/labels/train/{Path(label).name}")
        elif label_stem in [Path(p).stem for p in val_imgs]:
            shutil.move(label, f"{path}/labels/val/{Path(label).name}")
        elif label_stem in [Path(p).stem for p in test_imgs]:
            shutil.move(label, f"{path}/labels/test/{Path(label).name}")
        else:
            print(f"Warning: {label_stem} does not correspond to any image")

print(f"\n{'='*60}")
print("All 50 folders processed successfully!")
print(f"{'='*60}")