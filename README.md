# LineFormer Fine-tuned on Battery Discharge Curves

Fine-tuned version of [LineFormer](https://github.com/TheJaeLal/LineFormer) (ICDAR 2023) specialized for detecting line charts in battery charge/discharge curve figures from scientific papers.

**Pre-trained model weights**: [Hugging Face Hub](https://huggingface.co/t29mato/lineformer-battery-finetuned)

## Performance

Evaluated using the original LineFormer evaluation methodology (ICDAR 2023 Task 6a/6b), which matches predicted lines to ground truth lines using linear interpolation and the Hungarian algorithm.

| Model | Task 6a | Task 6b | GT Lines | Detected | Over-detection |
|-------|---------|---------|----------|----------|----------------|
| Pre-trained | **0.9471** | 0.6835 | 146 | 237 | +62.3% |
| Fine-tuned Best (iter_1300) | 0.9180 | 0.7394 | 146 | 179 | +22.6% |
| **Fine-tuned Final (iter_5000)** | 0.9097 | **0.7836** | **146** | **160** | **+9.6%** |

- **Task 6a**: Measures how well each GT line is matched (no penalty for extra detections)
- **Task 6b**: Penalizes over-detection — **the more practical metric**

**Key improvements (Pre-trained → Fine-tuned iter_5000):**
- Task 6b: 0.6835 → 0.7836 (+14.6%)
- Over-detection reduced from +62.3% to +9.6%
- Reduced false positives from text annotations, legends, and axis labels

## Setup

Based on [MMDetection](https://github.com/open-mmlab/mmdetection). Tested on PyTorch 2.1.0 + CUDA 12.1.

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
bash install.sh
```

## Inference

1. Download the fine-tuned checkpoint from [Hugging Face Hub](https://huggingface.co/t29mato/lineformer-battery-finetuned)
2. Run inference:

```python
import infer
import cv2
import line_utils

img_path = "your_battery_curve.png"
img = cv2.imread(img_path)

CKPT = "lineformer_battery_iter_5000.pth"
CONFIG = "lineformer_swin_t_config.py"
DEVICE = "cuda:0"

infer.load_model(CONFIG, CKPT, DEVICE)
line_dataseries = infer.get_dataseries(img, to_clean=False)

# Visualize extracted line keypoints
img = line_utils.draw_lines(img, line_utils.points_to_array(line_dataseries))
cv2.imwrite("result.png", img)
```

## Model Files

Two checkpoints are available on [Hugging Face Hub](https://huggingface.co/t29mato/lineformer-battery-finetuned):

| Model | Task 6b | segm_mAP | Over-detection | Best For |
|-------|---------|----------|----------------|----------|
| **lineformer_battery_iter_5000.pth** | **0.7836** | 0.1451 | **+9.6%** | **Data extraction accuracy** (recommended) |
| lineformer_battery_best_iter_1300.pth | 0.7394 | **0.1587** | +22.6% | Mask shape precision |

## Evaluation

Evaluate using the original paper's metric6a/6b:

```bash
python scripts/eval_metric6a_battery.py --compare-all
```

## Fine-tuning

```bash
python lineformer_finetune_battery_gpu.py
```

### Training Details

- **Optimizer**: SGD (lr=0.0001, momentum=0.9, weight_decay=0.0001)
- **LR Schedule**: Step decay at iterations [3500, 4750]
- **Total Iterations**: 5000
- **Batch Size**: 2
- **Backbone**: ResNet-50 (frozen, pre-trained on ImageNet)
- **Train**: 62 images, **Validation**: 19 images
- **Format**: COCO instance segmentation

## Limitations

- **Dense multi-cycle curves** (e.g., 30-cycle plots): GT=4 lines → Detected=14-17 lines
- **Dashed lines and annotations**: Non-line elements may be falsely detected
- **Mask IoU precision**: Low mAP_75 suggests mask boundary accuracy needs improvement

## Project Structure

```
├── lineformer_finetune_battery_gpu.py  # Fine-tuning script
├── lineformer_swin_t_config.py         # Model config
├── infer.py                            # Inference module
├── eval.py                             # Original evaluation
├── metric6a.py                         # ICDAR Task 6a/6b metrics
├── scripts/
│   ├── eval_metric6a_battery.py        # Battery-specific evaluation
│   ├── eval_val_dataset.py             # Validation dataset evaluation
│   ├── wpd_to_coco.py                  # WPD → COCO format conversion
│   └── filter_wpd.py                   # Data filtering
├── data_processing/
│   ├── COCO_Converter.ipynb            # COCO format converter
│   └── DataFormat.ipynb                # Data format exploration
└── reports/battery_finetune/           # Fine-tuning results report
```

## Citation

```bibtex
@InProceedings{10.1007/978-3-031-41734-4_24,
  author="Lal, Jay and Mitkari, Aditya and Bhosale, Mahesh and Doermann, David",
  title="LineFormer: Line Chart Data Extraction Using Instance Segmentation",
  booktitle="Document Analysis and Recognition - ICDAR 2023",
  year="2023",
  publisher="Springer Nature Switzerland",
  pages="387--400"
}
```

## License

Apache 2.0
