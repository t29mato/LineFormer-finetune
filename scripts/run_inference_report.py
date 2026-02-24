"""Run inference on a sample image with multiple checkpoints and generate overlay visualizations.

Usage:
    python scripts/run_inference_report.py \
        --image data/sample_figure/10.3390_nano14040384_4i.png \
        --config lineformer_finetune_battery_gpu.py \
        --output-dir reports/battery_finetune
"""

import argparse
import itertools
import os
import sys

import cv2
import mmcv
import numpy as np
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
    partition = 1.0 / (n + 1)
    return [hsv_to_bgr(partition * i, 1.0, 1.0) for i in range(n)]


def parse_result(result, score_thresh=0.3):
    bbox, masks = result[0][0], result[1][0]
    scores = bbox[:, 4]
    inst_masks = [m for m, s in zip(masks, scores) if s > score_thresh]
    inst_scores = [float(s) for s in scores if s > score_thresh]
    return inst_masks, inst_scores


def create_overlay(img, masks, alpha=0.5):
    """Draw detected line masks as colored overlays on the original image."""
    overlay = img.copy()
    if not masks:
        return overlay

    colors = get_distinct_colors(len(masks))
    for idx, mask in enumerate(masks):
        color = colors[idx]
        # Create colored mask
        colored = np.zeros_like(overlay)
        colored[:] = color
        mask_bool = mask.astype(bool)
        overlay[mask_bool] = cv2.addWeighted(
            overlay[mask_bool], 1 - alpha,
            colored[mask_bool], alpha, 0
        )
        # Draw thin contour for clarity
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 1)

    return overlay


def run_single_checkpoint(config_path, ckpt_path, img_path, device, score_thr=0.3):
    """Run inference with a single checkpoint and return masks + scores."""
    model = init_detector(config_path, ckpt_path, device=device)
    result = inference_detector(model, img_path)
    masks, scores = parse_result(result, score_thr)
    del model
    import torch
    torch.cuda.empty_cache()
    return masks, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='reports/battery_finetune')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--score-thr', type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, 'images'), exist_ok=True)

    # Define checkpoints to evaluate
    checkpoints = [
        ('Pre-trained (iter_3000)', 'checkpoints/models/iter_3000.pth'),
        ('Fine-tuned best (iter_1300)', 'work_dirs/battery_finetune/best_segm_mAP_iter_1300.pth'),
        ('Fine-tuned iter_4000', 'work_dirs/battery_finetune/iter_4000.pth'),
        ('Fine-tuned iter_5000', 'work_dirs/battery_finetune/iter_5000.pth'),
    ]

    img = mmcv.imread(args.image)
    # Save original
    cv2.imwrite(os.path.join(args.output_dir, 'images', 'original.png'), img)

    results_summary = []

    for label, ckpt_path in checkpoints:
        if not os.path.exists(ckpt_path):
            print(f'SKIP: {ckpt_path} not found')
            continue

        print(f'\n=== {label} ({ckpt_path}) ===')
        masks, scores = run_single_checkpoint(
            args.config, ckpt_path, args.image, args.device, args.score_thr
        )
        print(f'  Detected {len(masks)} lines, scores: {[f"{s:.3f}" for s in scores]}')

        # Create overlay
        overlay = create_overlay(img, masks, alpha=0.45)

        # Add label text
        cv2.putText(overlay, f'{label} ({len(masks)} lines)',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(overlay, f'{label} ({len(masks)} lines)',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        safe_label = label.replace(' ', '_').replace('(', '').replace(')', '')
        out_path = os.path.join(args.output_dir, 'images', f'{safe_label}.png')
        cv2.imwrite(out_path, overlay)
        print(f'  Saved: {out_path}')

        results_summary.append({
            'label': label,
            'ckpt': ckpt_path,
            'num_lines': len(masks),
            'scores': scores,
            'image_file': f'images/{safe_label}.png',
        })

    # Save summary JSON
    import json
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f'\nDone. Results saved to {args.output_dir}/')


if __name__ == '__main__':
    main()
