"""Filter WPD tar files to identify charge-discharge curves for battery materials.

Charge-discharge curves typically have:
- Y-axis: Voltage (V), range ~1.0-6.0V
- X-axis: Capacity (mAh/g, mAh, Ah/kg), values > 50
- Multiple datasets (charge + discharge curves)

Usage:
    python scripts/filter_wpd.py --data-dir data/WPD_file --output scripts/manifest.json
"""

import argparse
import json
import os
import tarfile
from pathlib import Path


def load_wpd_from_tar(tar_path):
    """Extract and parse wpd.json from a tar file."""
    with tarfile.open(tar_path, 'r') as tf:
        for member in tf.getmembers():
            if member.name.endswith('wpd.json'):
                f = tf.extractfile(member)
                if f:
                    return json.load(f)
    return None


def is_charge_discharge(wpd_data):
    """Determine if a WPD dataset represents a charge-discharge curve.

    Returns (bool, dict) where dict contains diagnostic info.
    """
    datasets = wpd_data.get('datasetColl', [])
    non_empty = [ds for ds in datasets if ds.get('data')]

    if len(non_empty) < 2:
        return False, {'reason': f'too few datasets ({len(non_empty)})'}

    all_ys = []
    all_xs = []
    for ds in non_empty:
        for pt in ds['data']:
            val = pt.get('value', [])
            if len(val) >= 2:
                all_xs.append(val[0])
                all_ys.append(val[1])

    if not all_ys or not all_xs:
        return False, {'reason': 'no valid data points'}

    y_min, y_max = min(all_ys), max(all_ys)
    x_min, x_max = min(all_xs), max(all_xs)

    # Charge-discharge: voltage on Y-axis (1.0-6.0V), capacity on X-axis (> 50)
    if not (1.0 <= y_min and y_max <= 6.0):
        return False, {
            'reason': f'Y range [{y_min:.1f}, {y_max:.1f}] outside voltage range [1.0, 6.0]V'
        }

    if x_max <= 50:
        return False, {
            'reason': f'X max {x_max:.1f} too small for capacity'
        }

    info = {
        'y_range': [round(y_min, 2), round(y_max, 2)],
        'x_range': [round(x_min, 2), round(x_max, 2)],
        'num_datasets': len(non_empty),
        'total_points': len(all_ys),
    }
    return True, info


EXCLUDE_FILES = {
    'SID51009_Fig4d',
    'SID51009_Fig4e',
    'SID51009_Fig4f',
    'SID51009_Fig4g',
    'SID51009_Fig4h',
}


def main():
    parser = argparse.ArgumentParser(description='Filter WPD files for charge-discharge curves')
    parser.add_argument('--data-dir', type=str, default='data/WPD_file',
                        help='Path to WPD_file directory')
    parser.add_argument('--output', type=str, default='scripts/manifest.json',
                        help='Output manifest JSON path')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tar_files = sorted(data_dir.glob('*/*.tar'))

    print(f'Found {len(tar_files)} tar files')

    accepted = []
    rejected = []

    for tar_path in tar_files:
        wpd_data = load_wpd_from_tar(str(tar_path))
        if wpd_data is None:
            rejected.append({
                'path': str(tar_path),
                'reason': 'failed to load wpd.json'
            })
            continue

        is_cd, info = is_charge_discharge(wpd_data)
        entry = {
            'path': str(tar_path),
            'name': tar_path.stem,
            'sid': tar_path.parent.name,
        }
        entry.update(info)

        if tar_path.stem in EXCLUDE_FILES:
            entry['reason'] = 'partial annotation (excluded manually)'
            rejected.append(entry)
        elif is_cd:
            accepted.append(entry)
        else:
            rejected.append(entry)

    print(f'\nResults:')
    print(f'  Charge-discharge curves: {len(accepted)}')
    print(f'  Rejected: {len(rejected)}')

    if rejected:
        print(f'\nRejected files:')
        for r in rejected:
            print(f"  {r['path']}: {r.get('reason', 'unknown')}")

    total_lines = sum(e.get('num_datasets', 0) for e in accepted)
    print(f'\nTotal line annotations: {total_lines}')
    print(f'Average lines per image: {total_lines / len(accepted):.1f}' if accepted else '')

    manifest = {
        'accepted': accepted,
        'rejected': rejected,
        'stats': {
            'total_files': len(tar_files),
            'accepted_count': len(accepted),
            'rejected_count': len(rejected),
            'total_line_annotations': total_lines,
        }
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'\nManifest saved to {args.output}')


if __name__ == '__main__':
    main()
