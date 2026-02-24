"""Convert filtered WPD tar files to COCO format for LineFormer training.

Reads the manifest from filter_wpd.py and converts accepted charge-discharge
curve data to COCO instance segmentation format.

Usage:
    python scripts/wpd_to_coco.py \
        --manifest scripts/manifest.json \
        --output-dir data/coco_battery \
        --val-ratio 0.2
"""

import argparse
import copy
import json
import os
import tarfile
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util
from bresenham import bresenham


# --- Helper functions (from COCO_Converter.ipynb) ---

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def padd_square(img, desired_size, padd_color=255):
    if padd_color == 255 and img.ndim == 3:
        padd_color = [255, 255, 255]
    size = img.shape[:2]
    delta_w = desired_size - size[1]
    delta_h = desired_size - size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=padd_color)
    return new_img, (left, top)


def resize_image(img, max_size=512, padd=True):
    clean_img = img
    tx_crop, ty_crop = 0, 0
    sx, sy = 1, 1
    tx_padd, ty_padd = 0, 0

    h_cropped, w_cropped = clean_img.shape[:2]
    if max_size:
        if clean_img.shape[0] > clean_img.shape[1]:
            clean_img = resize(clean_img, height=max_size)
        else:
            clean_img = resize(clean_img, width=max_size)
        sx = float(clean_img.shape[1]) / w_cropped
        sy = float(clean_img.shape[0]) / h_cropped

    if padd:
        clean_img, (tx_padd, ty_padd) = padd_square(clean_img, max_size)

    transformation = (sx, sy, tx_crop, ty_crop, tx_padd, ty_padd)
    return clean_img, transformation


def transform_line_data(lines_data, transformation):
    lines_data = copy.deepcopy(lines_data)
    (sx, sy, tx_crop, ty_crop, tx_padd, ty_padd) = transformation

    def _transform_x(x):
        return int((x - tx_crop) * sx + tx_padd)

    def _transform_y(y):
        return int((y - ty_crop) * sy + ty_padd)

    for ln in lines_data:
        for pt in ln:
            pt['x'] = _transform_x(pt['x'])
            pt['y'] = _transform_y(pt['y'])
    return lines_data


def get_lines(ln_data):
    pt = []
    ipt = []
    for ln in ln_data:
        ptl = []
        iptl = []
        for ix, pt_ in enumerate(ln):
            px = pt_['x']
            py = pt_['y']
            ptl.append(list((px, py)))
            if ix > 0:
                ppx, ppy = ln[ix - 1]['x'], ln[ix - 1]['y']
                iptl.append(bresenham(ppx, ppy, px, py))
        pt.append(ptl)
        ipt.append(iptl)
    return pt, ipt


def get_inst_masks_bin(img_h, img_w, ln_data, thickness=1):
    all_points, all_ipoints = get_lines(ln_data)
    all_inst_masks = []

    for l_idx, l_ in enumerate(all_ipoints):
        l_all_pts = [pt for pt_set in l_ for pt in pt_set]
        inst_mask = np.zeros((img_w, img_h))
        for (x, y) in l_all_pts:
            x = min(max(0, x), img_w - 1)
            y = min(max(0, y), img_h - 1)
            inst_mask[int(x), int(y)] = 1

        if thickness > 1:
            inst_mask = cv2.dilate(
                inst_mask,
                cv2.getStructuringElement(cv2.MORPH_RECT, (thickness, thickness))
            )
        all_inst_masks.append(inst_mask.T)

    return all_inst_masks


# --- WPD-specific functions ---

def parse_wpd_lines(wpd_data):
    """Convert WPD datasets to lines_point_data format (pixel coordinates)."""
    lines = []
    for ds in wpd_data.get('datasetColl', []):
        if ds.get('data'):
            line = [{'x': int(round(pt['x'])), 'y': int(round(pt['y']))}
                    for pt in ds['data']]
            if len(line) >= 2:
                lines.append(line)
    return lines


def load_image_from_tar(tar_path):
    """Extract image.png from tar as numpy array."""
    with tarfile.open(tar_path, 'r') as tf:
        for member in tf.getmembers():
            if member.name.endswith('image.png'):
                f = tf.extractfile(member)
                if f:
                    img_bytes = f.read()
                    img_array = cv2.imdecode(
                        np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR
                    )
                    return img_array
    return None


def load_wpd_from_tar(tar_path):
    """Extract wpd.json from tar."""
    with tarfile.open(tar_path, 'r') as tf:
        for member in tf.getmembers():
            if member.name.endswith('wpd.json'):
                f = tf.extractfile(member)
                if f:
                    return json.load(f)
    return None


def process_single_image(sample_idx, tar_path, save_img_path, line_thickness=3):
    """Process one tar file into COCO format annotation."""
    img = load_image_from_tar(tar_path)
    wpd_data = load_wpd_from_tar(tar_path)

    if img is None or wpd_data is None:
        raise ValueError(f'Failed to load data from {tar_path}')

    lines_point_data = parse_wpd_lines(wpd_data)
    if not lines_point_data:
        raise ValueError(f'No valid line data in {tar_path}')

    # Resize to 512x512
    clean_img, transformation = resize_image(img, max_size=512, padd=True)
    h, w = clean_img.shape[:2]

    # Transform line coordinates
    lines_point_data = transform_line_data(lines_point_data, transformation)

    # Generate instance masks
    line_masks = get_inst_masks_bin(h, w, lines_point_data, line_thickness)

    # Save resized image
    if save_img_path:
        os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
        cv2.imwrite(str(save_img_path), clean_img)

    # Build COCO annotation
    file_name = Path(tar_path).stem + '.png'
    img_info = dict(id=sample_idx, file_name=file_name, height=h, width=w)

    annotations = []
    for m_idx, mask in enumerate(line_masks):
        segment_mask = mask > 0
        rle_mask = mask_util.encode(
            np.array(segment_mask[:, :, np.newaxis], order='F', dtype='uint8')
        )[0]
        rle_mask['counts'] = rle_mask['counts'].decode()
        annotations.append({
            'image_id': img_info['id'],
            'category_id': 1,
            'iscrowd': 0,
            'bbox': [1, 2, 3, 4],  # Placeholder (Mask2Former uses masks, not bboxes)
            'area': 10,             # Non-zero required by MMDetection COCO loader
            'segmentation': rle_mask,
        })

    return img_info, annotations


def split_by_sid(accepted_entries, val_ratio=0.2):
    """Split entries by SID to avoid data leakage between train/val."""
    sids = sorted(set(e['sid'] for e in accepted_entries))
    # Calculate cumulative image count per SID
    sid_counts = {}
    for e in accepted_entries:
        sid_counts[e['sid']] = sid_counts.get(e['sid'], 0) + 1

    total = len(accepted_entries)
    val_target = int(total * val_ratio)

    # Assign SIDs to val from the end until we reach target
    val_sids = set()
    val_count = 0
    for sid in reversed(sids):
        if val_count >= val_target:
            break
        val_sids.add(sid)
        val_count += sid_counts[sid]

    train_entries = [e for e in accepted_entries if e['sid'] not in val_sids]
    val_entries = [e for e in accepted_entries if e['sid'] in val_sids]

    return train_entries, val_entries


def main():
    parser = argparse.ArgumentParser(description='Convert WPD data to COCO format')
    parser.add_argument('--manifest', type=str, default='scripts/manifest.json',
                        help='Path to filter manifest')
    parser.add_argument('--output-dir', type=str, default='data/coco_battery',
                        help='Output COCO dataset directory')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Validation set ratio')
    parser.add_argument('--line-thickness', type=int, default=3,
                        help='Line mask thickness in pixels')
    args = parser.parse_args()

    with open(args.manifest, 'r') as f:
        manifest = json.load(f)

    accepted = manifest['accepted']
    print(f'Processing {len(accepted)} charge-discharge curve files')

    # Split by SID
    train_entries, val_entries = split_by_sid(accepted, args.val_ratio)
    print(f'Train: {len(train_entries)} images, Val: {len(val_entries)} images')

    train_sids = sorted(set(e['sid'] for e in train_entries))
    val_sids = sorted(set(e['sid'] for e in val_entries))
    print(f'Train SIDs ({len(train_sids)}): {train_sids}')
    print(f'Val SIDs ({len(val_sids)}): {val_sids}')

    categories = [{
        'supercategory': 'foreground',
        'color': [220, 20, 60],
        'isthing': 1,
        'id': 1,
        'name': 'line',
    }]

    output_dir = Path(args.output_dir)

    for split_name, entries in [('train', train_entries), ('val', val_entries)]:
        img_dir = output_dir / f'{split_name}_images'
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(output_dir / 'annotations', exist_ok=True)

        all_images = []
        all_annotations = []
        errors = []

        for idx, entry in enumerate(entries):
            tar_path = entry['path']
            save_img_path = img_dir / (Path(tar_path).stem + '.png')

            try:
                img_info, annots = process_single_image(
                    idx, tar_path, str(save_img_path), args.line_thickness
                )
                all_images.append(img_info)
                all_annotations.extend(annots)
                print(f'  [{split_name}] {idx + 1}/{len(entries)}: '
                      f'{Path(tar_path).stem} - {len(annots)} lines')
            except Exception as e:
                errors.append({'path': tar_path, 'error': str(e)})
                print(f'  [{split_name}] ERROR: {tar_path}: {e}')

        # Assign unique annotation IDs
        for ann_idx, ann in enumerate(all_annotations):
            ann['id'] = ann_idx + 1

        coco_data = {
            'images': all_images,
            'annotations': all_annotations,
            'categories': categories,
        }

        annot_path = output_dir / 'annotations' / f'{split_name}_coco_annot.json'
        with open(annot_path, 'w') as f:
            json.dump(coco_data, f)

        print(f'\n{split_name}: {len(all_images)} images, '
              f'{len(all_annotations)} annotations')
        print(f'Saved to {annot_path}')

        if errors:
            print(f'Errors ({len(errors)}):')
            for e in errors:
                print(f"  {e['path']}: {e['error']}")

        print()


if __name__ == '__main__':
    main()
