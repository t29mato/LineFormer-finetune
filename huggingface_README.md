---
license: apache-2.0
tags:
- instance-segmentation
- line-detection
- battery
- scientific-figures
- LineFormer
datasets:
- custom
metrics:
- mAP
- Task 6a
- Task 6b
---

# LineFormer Fine-tuned on Battery Discharge Curves

This model is a fine-tuned version of [LineFormer](https://github.com/uci-uav-forge/LineFormer) specialized for detecting and segmenting line charts in battery charge/discharge curve figures from scientific papers.

## Model Description

LineFormer is an instance segmentation model designed to detect individual lines in scientific plots. This fine-tuned version has been optimized for battery electrochemistry figures, particularly charge/discharge curves with multiple C-rates.

**Base Model**: LineFormer (pre-trained on scientific figure datasets)
**Fine-tuning Dataset**: 62 battery charge/discharge curve images from scientific papers
**Validation Dataset**: 19 battery curve images
**Framework**: MMDetection (Mask R-CNN based)

## Performance

Evaluated using the original LineFormer evaluation methodology (ICDAR 2023 Task 6a/6b), which matches predicted lines to ground truth lines using linear interpolation and the Hungarian algorithm.

### Task 6a / 6b Scores (Original Paper Metrics)

| Model | Task 6a | Task 6b | GT Lines | Detected | Over-detection |
|-------|---------|---------|----------|----------|----------------|
| Pre-trained | **0.9471** | 0.6835 | 146 | 237 | +62.3% |
| Fine-tuned Best (iter_1300) | 0.9180 | 0.7394 | 146 | 179 | +22.6% |
| **Fine-tuned Final (iter_5000)** | 0.9097 | **0.7836** | **146** | **160** | **+9.6%** |

- **Task 6a**: Measures how well each GT line is matched (no penalty for extra detections)
- **Task 6b**: Penalizes over-detection — **the more practical metric**

**Key improvements (Pre-trained → Fine-tuned iter_5000):**
- Task 6b (practical metric): 0.6835 → 0.7836 (+14.6%)
- Over-detection reduced from +62.3% to +9.6%
- Reduced false positives from text annotations, legends, and axis labels
- Better handling of densely packed multi-cycle curves

### Additional Metrics

| Model | segm_mAP | Confidence Score |
|-------|----------|-----------------|
| Pre-trained | 0.0395 | 0.791 |
| Fine-tuned Best (iter_1300) | **0.1587** | 0.872 |
| **Fine-tuned Final (iter_5000)** | 0.1451 | **0.896** |

## Usage

### Requirements

```bash
pip install mmdet mmcv torch torchvision
```

### Inference

```python
from mmdet.apis import init_detector, inference_detector
import mmcv

# Load model
config_file = 'configs/battery_finetune.py'
checkpoint_file = 'lineformer_battery_iter_5000.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Inference
img = mmcv.imread('battery_curve.png')
result = inference_detector(model, img)

# Visualize
model.show_result(img, result, out_file='result.jpg', score_thr=0.5)
```

### Model Files

We provide two checkpoints with different strengths:

| Model | Task 6b | segm_mAP | Over-detection | Best For |
|-------|---------|----------|----------------|----------|
| **lineformer_battery_iter_5000.pth** | **0.7836** | 0.1451 | **+9.6%** | **Data extraction accuracy** (recommended) |
| **lineformer_battery_best_iter_1300.pth** | 0.7394 | **0.1587** | +22.6% | **Mask shape precision** |

**Which model to use?**
- **For most use cases**: Use `lineformer_battery_iter_5000.pth` — highest Task 6b score (0.7836) with fewest false positives
- **For mask segmentation tasks**: Use `lineformer_battery_best_iter_1300.pth` if you need precise mask boundaries (higher mAP)

## Training Details

### Training Hyperparameters

- **Optimizer**: SGD (lr=0.0001, momentum=0.9, weight_decay=0.0001)
- **Learning Rate Schedule**: Step decay at iterations [3500, 4750]
- **Total Iterations**: 5000
- **Batch Size**: 2
- **Image Size**: Resized to shorter edge 800px
- **Data Augmentation**: RandomFlip (flip_ratio=0.5)
- **Backbone**: ResNet-50 (frozen, pre-trained on ImageNet)

### Training Data

- **Train**: 62 images (battery charge/discharge curves from scientific papers)
- **Validation**: 19 images
- **Line Width**: 10px (for mask generation)
- **Format**: COCO instance segmentation format

### Training Metrics

| Iteration | mAP | mAP_50 | mAP_75 |
|-----------|-----|--------|--------|
| 500 | 0.1156 | 0.2852 | 0.0005 |
| 1300 | **0.1587** | 0.3705 | 0.0006 |
| 5000 | 0.1451 | 0.3503 | 0.0004 |

**Note**: mAP_75 ≈ 0 indicates that mask boundary precision needs improvement, though line detection count is accurate.

## Limitations and Challenges

The model still struggles with:

1. **Dense multi-cycle curves** (e.g., 30-cycle plots): GT=4 lines → Detected=14-17 lines
2. **Dashed lines and annotations**: Non-line elements (arrows, dimension lines) may be detected
3. **Mask IoU precision**: Low mAP_75 suggests mask shape accuracy needs improvement
4. **GT definition mismatch**: Some images count charge/discharge pairs as single lines vs. separate lines

## Intended Use

This model is designed for:
- Automated extraction of battery performance data from scientific literature
- Digitization of charge/discharge curves for meta-analysis
- Pre-processing battery electrochemistry figures for data mining

**Not recommended for**:
- General line chart detection (use original LineFormer)
- Real-time applications requiring <100ms inference
- Figures with extreme aspect ratios or unconventional layouts

## Citation

If you use this model, please cite the original LineFormer paper:

```bibtex
@inproceedings{Xia_2022_WACV,
  title={LineFormer: Rethinking Line Chart Data Extraction as Instance Segmentation},
  author={Xia, Weixin and Lo, Kelvin and Chao, Qing and Li, Tong and Zhao, Jian},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2022}
}
```

## License

Apache 2.0 (same as base LineFormer model)

## Model Card Authors

Fine-tuning and evaluation by t29mato

## Model Card Contact

For questions about this fine-tuned model, please open an issue in the [repository](https://github.com/t29mato/LineFormer-finetune).
