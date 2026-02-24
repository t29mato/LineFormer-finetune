"""Generate training curve plots for the report."""
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOG_FILE = 'work_dirs/battery_finetune/20260220_180100.log'
OUTPUT_DIR = 'reports/battery_finetune/images'

def parse_log():
    iters_loss = []
    losses = []
    iters_map = []
    maps = []
    maps_50 = []

    with open(LOG_FILE) as f:
        for line in f:
            # Training loss
            m = re.search(r'Iter \[(\d+)/5000\].*loss: ([0-9.]+)', line)
            if m:
                iters_loss.append(int(m.group(1)))
                losses.append(float(m.group(2)))
            # Validation mAP
            m = re.search(r'Iter\(val\).*segm_mAP: ([0-9.]+), segm_mAP_50: ([0-9.]+)', line)
            if m:
                maps.append(float(m.group(1)))
                maps_50.append(float(m.group(2)))

    iters_map = list(range(100, 100 * len(maps) + 1, 100))
    return iters_loss, losses, iters_map, maps, maps_50


def main():
    iters_loss, losses, iters_map, maps, maps_50 = parse_log()

    # --- Plot 1: Training Loss ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters_loss, losses, color='#e74c3c', linewidth=1.2, alpha=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training Loss over Iterations')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=2000, color='gray', linestyle='--', alpha=0.5, label='LR step (2000)')
    ax.axvline(x=4000, color='gray', linestyle=':', alpha=0.5, label='LR step (4000)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/training_loss.png', dpi=150)
    plt.close()

    # --- Plot 2: Validation mAP ---
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(iters_map, maps, color='#2ecc71', linewidth=1.5, marker='o', markersize=2, label='segm_mAP')
    ax1.plot(iters_map, maps_50, color='#3498db', linewidth=1.5, marker='s', markersize=2, label='segm_mAP_50')
    best_idx = np.argmax(maps)
    ax1.axvline(x=iters_map[best_idx], color='#e67e22', linestyle='--', alpha=0.7,
                label=f'Best mAP={maps[best_idx]:.4f} @ iter {iters_map[best_idx]}')
    ax1.scatter([iters_map[best_idx]], [maps[best_idx]], color='#e67e22', s=80, zorder=5, marker='*')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('mAP')
    ax1.set_title('Validation Segmentation mAP over Iterations')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/validation_map.png', dpi=150)
    plt.close()

    print(f'Plots saved to {OUTPUT_DIR}/')
    print(f'  Best mAP: {maps[best_idx]:.4f} at iter {iters_map[best_idx]}')
    print(f'  Final mAP: {maps[-1]:.4f} at iter {iters_map[-1]}')
    print(f'  Final loss: {losses[-1]:.4f}')


if __name__ == '__main__':
    main()
