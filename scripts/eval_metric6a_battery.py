"""Evaluate battery fine-tuned LineFormer using original metric6a (ICDAR 2023).

Uses the same evaluation methodology as the original LineFormer paper:
  1. Model inference → masks → data point extraction (infer.get_dataseries)
  2. GT line data from WPD tar files → transform to 512x512 coords
  3. Compare using metric6a (linear interpolation + Hungarian algorithm)

Usage:
    python scripts/eval_metric6a_battery.py \
        --config lineformer_swin_t_config.py \
        --checkpoint checkpoints/models/iter_3000.pth \
        --manifest scripts/manifest.json \
        --data-dir data/coco_battery \
        --device cuda:0

    # Compare all 3 models:
    python scripts/eval_metric6a_battery.py --compare-all
"""

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np
import mmcv
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import infer
import metric6a
from scripts.wpd_to_coco import (
    load_image_from_tar,
    load_wpd_from_tar,
    parse_wpd_lines,
    resize_image,
    transform_line_data,
)


def get_val_entries(manifest_path, val_images_dir):
    """Get manifest entries that correspond to val images."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    val_names = {Path(p).stem for p in os.listdir(val_images_dir)}
    val_entries = [e for e in manifest['accepted'] if e['name'] in val_names]
    return val_entries


def get_gt_lines_transformed(tar_path):
    """Load GT lines from WPD tar and transform to 512x512 image coords."""
    img = load_image_from_tar(tar_path)
    wpd_data = load_wpd_from_tar(tar_path)

    if img is None or wpd_data is None:
        raise ValueError(f'Failed to load data from {tar_path}')

    lines_point_data = parse_wpd_lines(wpd_data)
    if not lines_point_data:
        raise ValueError(f'No valid line data in {tar_path}')

    # Transform to 512x512 (same as wpd_to_coco.py)
    _, transformation = resize_image(img, max_size=512, padd=True)
    lines_transformed = transform_line_data(lines_point_data, transformation)

    return lines_transformed


def evaluate_model(config_path, checkpoint_path, manifest_path, data_dir,
                   device='cuda:0'):
    """Evaluate a single model checkpoint using metric6a and 6b."""
    val_images_dir = Path(data_dir) / 'val_images'
    val_entries = get_val_entries(manifest_path, str(val_images_dir))

    print(f'Model: {checkpoint_path}')
    print(f'Val images: {len(val_entries)}')

    # Load model
    infer.load_model(config_path, checkpoint_path, device)

    results = []
    for entry in val_entries:
        name = entry['name']
        img_path = str(val_images_dir / f'{name}.png')
        img = mmcv.imread(img_path)

        # Model inference → data points
        try:
            pred_ds = infer.get_dataseries(
                img, annot=None, to_clean=False, post_proc=False,
                mask_kp_sample_interval=10
            )
        except Exception as e:
            print(f'  ERROR inference {name}: {e}')
            pred_ds = []

        # GT lines from WPD
        try:
            gt_ds = get_gt_lines_transformed(entry['path'])
        except Exception as e:
            print(f'  ERROR loading GT {name}: {e}')
            continue

        # Compute metric6a and 6b scores
        try:
            score_6a = metric6a.metric_6a_indv(pred_ds, gt_ds, 'line')
        except Exception as e:
            print(f'  ERROR metric6a {name}: {e}')
            score_6a = 0.0

        try:
            score_6b = metric6a.metric_6b_indv(pred_ds, gt_ds, 'line')
        except Exception as e:
            print(f'  ERROR metric6b {name}: {e}')
            score_6b = 0.0

        results.append({
            'name': name,
            'score_6a': score_6a,
            'score_6b': score_6b,
            'num_gt': len(gt_ds),
            'num_pred': len(pred_ds),
        })
        print(f'  {name}: 6a={score_6a:.4f}, 6b={score_6b:.4f} '
              f'(GT={len(gt_ds)}, Pred={len(pred_ds)})')

    df = pd.DataFrame(results)
    avg_6a = df['score_6a'].mean()
    avg_6b = df['score_6b'].mean()
    print(f'\n  Average 6a: {avg_6a:.4f}')
    print(f'  Average 6b: {avg_6b:.4f}')
    print(f'  Total GT lines: {df["num_gt"].sum()}, '
          f'Total Pred lines: {df["num_pred"].sum()}')

    return df


def compare_all_models(config_path, manifest_path, data_dir, device='cuda:0'):
    """Compare pre-trained, best, and final fine-tuned models."""
    models = {
        'Pre-trained (iter_3000)': 'checkpoints/models/iter_3000.pth',
        'Fine-tuned Best (iter_1300)': 'work_dirs/battery_finetune/best_segm_mAP_iter_1300.pth',
        'Fine-tuned Final (iter_5000)': 'work_dirs/battery_finetune/iter_5000.pth',
    }

    all_results = {}
    for model_name, ckpt_path in models.items():
        if not Path(ckpt_path).exists():
            print(f'\nSkipping {model_name}: {ckpt_path} not found')
            continue

        print(f'\n{"="*60}')
        print(f'Evaluating: {model_name}')
        print(f'{"="*60}')

        df = evaluate_model(config_path, ckpt_path, manifest_path, data_dir,
                            device)
        df['model'] = model_name
        all_results[model_name] = df

    # Summary comparison
    print(f'\n{"="*60}')
    print('SUMMARY COMPARISON')
    print(f'{"="*60}')
    print(f'{"Model":<35} {"6a":>8} {"6b":>8} {"GT":>6} {"Pred":>6}')
    print('-' * 65)

    for model_name, df in all_results.items():
        print(f'{model_name:<35} '
              f'{df["score_6a"].mean():>8.4f} '
              f'{df["score_6b"].mean():>8.4f} '
              f'{df["num_gt"].sum():>6} '
              f'{df["num_pred"].sum():>6}')

    # Save combined results
    if all_results:
        combined = pd.concat(all_results.values(), ignore_index=True)
        output_path = 'reports/battery_finetune/metric6a_comparison.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined.to_csv(output_path, index=False)
        print(f'\nDetailed results saved to: {output_path}')

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate LineFormer on battery data using metric6a')
    parser.add_argument('--config', type=str,
                        default='lineformer_swin_t_config.py')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/models/iter_3000.pth')
    parser.add_argument('--manifest', type=str,
                        default='scripts/manifest.json')
    parser.add_argument('--data-dir', type=str,
                        default='data/coco_battery')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--compare-all', action='store_true',
                        help='Compare all 3 models')
    args = parser.parse_args()

    if args.compare_all:
        compare_all_models(args.config, args.manifest, args.data_dir,
                           args.device)
    else:
        evaluate_model(args.config, args.checkpoint, args.manifest,
                       args.data_dir, args.device)


if __name__ == '__main__':
    main()
