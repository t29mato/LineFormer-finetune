"""Evaluate multiple checkpoints on the full Val dataset.

Runs inference on each val image, generates overlay visualizations,
and computes per-image detection counts vs ground truth.

Usage:
    python scripts/eval_val_dataset.py \
        --config lineformer_finetune_battery_gpu.py \
        --output-dir reports/battery_finetune
"""

import argparse
import itertools
import json
import os
from collections import Counter
from pathlib import Path

import cv2
import mmcv
import numpy as np
import pycocotools.mask as mask_util
from mmdet.apis import inference_detector, init_detector


def hsv_to_bgr(h, s, v):
    c = v * s
    x = c * (1 - abs((h * 6) % 2 - 1))
    m = v - c
    if h < 1/6:
        r, g, b = c, x, 0
    elif h < 1/3:
        r, g, b = x, c, 0
    elif h < 0.5:
        r, g, b = 0, c, x
    elif h < 2/3:
        r, g, b = 0, x, c
    elif h < 5/6:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (int((b + m) * 255), int((g + m) * 255), int((r + m) * 255))


def get_distinct_colors(n):
    if n == 0:
        return []
    partition = 1.0 / (n + 1)
    return [hsv_to_bgr(partition * i, 1.0, 1.0) for i in range(n)]


def parse_result(result, score_thresh=0.3):
    bbox, masks = result[0][0], result[1][0]
    scores = bbox[:, 4]
    inst_masks = [m for m, s in zip(masks, scores) if s > score_thresh]
    inst_scores = [float(s) for s in scores if s > score_thresh]
    return inst_masks, inst_scores


def create_overlay(img, masks, alpha=0.45):
    overlay = img.copy()
    if not masks:
        return overlay
    colors = get_distinct_colors(len(masks))
    for idx, mask in enumerate(masks):
        color = colors[idx]
        colored = np.zeros_like(overlay)
        colored[:] = color
        mask_bool = mask.astype(bool)
        overlay[mask_bool] = cv2.addWeighted(
            overlay[mask_bool], 1 - alpha, colored[mask_bool], alpha, 0
        )
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 1)
    return overlay


def compute_mask_iou(pred_mask, gt_rle, h, w):
    """Compute IoU between a predicted boolean mask and a GT RLE mask."""
    pred_rle = mask_util.encode(
        np.array(pred_mask[:, :, np.newaxis], order='F', dtype='uint8')
    )[0]
    return float(mask_util.iou([pred_rle], [gt_rle], [0])[0][0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='reports/battery_finetune')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.3)
    args = parser.parse_args()

    # Load val annotation
    ann_file = 'data/coco_battery/annotations/val_coco_annot.json'
    img_prefix = 'data/coco_battery/val_images'
    with open(ann_file) as f:
        coco_data = json.load(f)

    gt_counts = Counter(a['image_id'] for a in coco_data['annotations'])

    # Checkpoints to evaluate
    checkpoints = [
        ('pretrained', 'Pre-trained (iter_3000)', 'checkpoints/models/iter_3000.pth'),
        ('best_iter1300', 'Fine-tuned Best (iter_1300)', 'work_dirs/battery_finetune/best_segm_mAP_iter_1300.pth'),
        ('iter5000', 'Fine-tuned (iter_5000)', 'work_dirs/battery_finetune/iter_5000.pth'),
    ]

    all_results = {}

    for ckpt_key, ckpt_label, ckpt_path in checkpoints:
        if not os.path.exists(ckpt_path):
            print(f'SKIP: {ckpt_path}')
            continue

        print(f'\n=== {ckpt_label} ===')
        model = init_detector(args.config, ckpt_path, device=args.device)

        img_out_dir = os.path.join(args.output_dir, 'images', f'val_{ckpt_key}')
        os.makedirs(img_out_dir, exist_ok=True)

        per_image_results = []

        for img_info in coco_data['images']:
            img_path = os.path.join(img_prefix, img_info['file_name'])
            img = mmcv.imread(img_path)

            result = inference_detector(model, img)
            masks, scores = parse_result(result, args.score_thr)

            gt_count = gt_counts.get(img_info['id'], 0)
            det_count = len(masks)
            avg_score = float(np.mean(scores)) if scores else 0.0

            per_image_results.append({
                'file_name': img_info['file_name'],
                'gt_lines': gt_count,
                'det_lines': det_count,
                'avg_score': round(avg_score, 3),
                'scores': [round(s, 3) for s in scores],
            })

            # Save overlay
            overlay = create_overlay(img, masks)
            out_path = os.path.join(img_out_dir, img_info['file_name'])
            cv2.imwrite(out_path, overlay)

            print(f'  {img_info["file_name"]}: GT={gt_count}, Det={det_count}, AvgScore={avg_score:.3f}')

        total_gt = sum(r['gt_lines'] for r in per_image_results)
        total_det = sum(r['det_lines'] for r in per_image_results)
        overall_avg_score = np.mean([r['avg_score'] for r in per_image_results if r['avg_score'] > 0])

        summary = {
            'label': ckpt_label,
            'ckpt': ckpt_path,
            'total_gt_lines': total_gt,
            'total_det_lines': total_det,
            'overall_avg_score': round(float(overall_avg_score), 3),
            'per_image': per_image_results,
        }
        all_results[ckpt_key] = summary

        print(f'\n  Total: GT={total_gt}, Det={total_det}, AvgScore={overall_avg_score:.3f}')

        del model
        import torch
        torch.cuda.empty_cache()

    # Save results
    with open(os.path.join(args.output_dir, 'val_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f'\nResults saved to {args.output_dir}/val_results.json')


if __name__ == '__main__':
    main()
