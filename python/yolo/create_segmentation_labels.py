#!/usr/bin/env python3
"""
Convert mask images to YOLO segmentation format (polygon points).
Generates polygon labels from mask_visib/ folder.

YOLO Segmentation Format:
  class_id x1_norm y1_norm x2_norm y2_norm ... xn_norm yn_norm
  (polygon points as normalized coordinates)
"""

import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image


def get_polygon_from_mask(mask_image, min_area=10):
    """
    Extract polygon contours from a mask image.
    
    Args:
        mask_image: Binary mask (0=background, >0=object)
        min_area: Minimum contour area to keep (to ignore noise)
    
    Returns:
        List of (x, y) coordinates forming the polygon, or None if no contour
    """
    # Find contours
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour (usually the object)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Filter by minimum area
    if cv2.contourArea(largest_contour) < min_area:
        return None
    
    # Simplify polygon (reduce number of points)
    epsilon = 0.002 * cv2.arcLength(largest_contour, True)
    simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Convert to list of (x, y) coordinates
    polygon = [(point[0][0], point[0][1]) for point in simplified]
    
    return polygon if len(polygon) >= 3 else None


def normalize_polygon(polygon, img_width, img_height):
    """
    Convert polygon coordinates to normalized format (0-1).
    
    Args:
        polygon: List of (x, y) coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        List of normalized coordinates (0-1)
    """
    normalized = []
    for x, y in polygon:
        norm_x = x / img_width
        norm_y = y / img_height
        # Clamp to [0, 1]
        norm_x = max(0, min(1, norm_x))
        norm_y = max(0, min(1, norm_y))
        normalized.append((norm_x, norm_y))
    
    return normalized


def create_segmentation_labels_for_folder(folder_path, verbose=False):
    """
    Generate segmentation labels for all scenes in an object folder.
    
    Args:
        folder_path: Path to object folder (e.g., train_pbr/000000)
        verbose: Print progress information
    
    Returns:
        Number of labels created
    """
    folder_path = Path(folder_path)
    
    # Load JSON files
    gt_file = folder_path / "scene_gt.json"
    gt_info_file = folder_path / "scene_gt_info.json"
    mask_dir = folder_path / "mask_visib"  # Use visible masks
    
    if not all([gt_file.exists(), gt_info_file.exists(), mask_dir.exists()]):
        if verbose:
            print(f"❌ Missing files in {folder_path.name}")
        return 0
    
    with open(gt_file, 'r') as f:
        scene_gt = json.load(f)
    
    with open(gt_info_file, 'r') as f:
        scene_gt_info = json.load(f)
    
    # Create labels directory
    labels_dir = folder_path / "labels_seg"
    labels_dir.mkdir(exist_ok=True)
    
    # Get image dimensions
    images_dir = folder_path / "rgb"
    sample_image_path = next(images_dir.glob("*.jpg"), None)
    if sample_image_path:
        sample_img = Image.open(sample_image_path)
        img_width, img_height = sample_img.size
    else:
        img_width, img_height = 640, 480  # Default
    
    label_count = 0
    
    # Process each scene
    for scene_id_str in sorted(scene_gt.keys(), key=lambda x: int(x)):
        scene_id = int(scene_id_str)
        objects = scene_gt[scene_id_str]
        
        # Skip empty scenes
        if not objects:
            # Create empty label file
            label_file = labels_dir / f"{scene_id:06d}.txt"
            label_file.write_text("")
            continue
        
        label_lines = []
        
        for obj_idx, obj_info in enumerate(objects):
            obj_id = obj_info['obj_id']
            class_id = obj_id - 1  # Convert to YOLO format (0-14)
            
            # Get visibility info
            obj_key = f"{scene_id:06d}_{obj_idx:06d}"
            if obj_key not in scene_gt_info:
                continue
            
            info = scene_gt_info[obj_key]
            visib_fract = info.get('visib_fract', 0.0)
            
            # Skip completely hidden objects
            if visib_fract <= 0.0:
                continue
            
            # Load mask
            mask_path = mask_dir / f"{scene_id:06d}_{obj_idx:06d}.png"
            if not mask_path.exists():
                continue
            
            mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_image is None:
                continue
            
            # Extract polygon from mask
            polygon = get_polygon_from_mask(mask_image)
            if polygon is None or len(polygon) < 3:
                continue
            
            # Normalize coordinates
            normalized_poly = normalize_polygon(polygon, img_width, img_height)
            
            # Format as YOLO segmentation label
            # class_id x1 y1 x2 y2 ... xn yn
            label_line = f"{class_id}"
            for x, y in normalized_poly:
                label_line += f" {x:.6f} {y:.6f}"
            
            label_lines.append(label_line)
        
        # Write label file
        label_file = labels_dir / f"{scene_id:06d}.txt"
        label_file.write_text("\n".join(label_lines))
        label_count += 1
        
        if verbose and (scene_id + 1) % 100 == 0:
            print(f"  Processed {scene_id + 1} scenes...")
    
    if verbose:
        print(f"  ✓ Created {label_count} label files")
    
    return label_count


def create_all_segmentation_labels(base_path="train_pbr", verbose=True):
    """
    Generate segmentation labels for all 50 object folders.
    
    Args:
        base_path: Path to train_pbr folder
        verbose: Print progress information
    
    Returns:
        Total number of labels created
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"❌ Base path not found: {base_path}")
        return 0
    
    object_folders = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()])
    
    print(f"\n{'='*70}")
    print(f"CREATING SEGMENTATION LABELS")
    print(f"{'='*70}")
    print(f"Processing {len(object_folders)} object folders...\n")
    
    total_labels = 0
    
    for i, folder in enumerate(object_folders, 1):
        print(f"{i:2d}/{len(object_folders)}: {folder.name}/ ", end="", flush=True)
        
        count = create_segmentation_labels_for_folder(folder, verbose=False)
        total_labels += count
        
        print(f"✓ {count} labels")
    
    print(f"\n{'='*70}")
    print(f"✓ Total labels created: {total_labels}")
    print(f"{'='*70}\n")
    
    return total_labels


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert masks to YOLO segmentation labels"
    )
    parser.add_argument(
        "--folder",
        default=None,
        help="Process single folder (e.g., train_pbr/000000). Default: process all"
    )
    parser.add_argument(
        "--base-path",
        default="train_pbr",
        help="Base path to train_pbr folder"
    )
    
    args = parser.parse_args()
    
    if args.folder:
        create_segmentation_labels_for_folder(args.folder, verbose=True)
    else:
        create_all_segmentation_labels(args.base_path, verbose=True)


if __name__ == "__main__":
    main()
