# LineFormer Fine-tuning Report: Battery Charge-Discharge Curves

**Date:** 2026-02-20
**Environment:** Ubuntu 24.04, NVIDIA GeForce RTX 4090 (24GB), Python 3.11, PyTorch 2.1.0+cu121

---

## 1. Overview

LineFormerは、線グラフからデータ系列を抽出するためのMask2Formerベースのインスタンスセグメンテーションモデルである（ICDAR 2023）。本報告書では、LineFormerの事前学習モデルを電池充放電曲線データセットでFine-tuningした結果をまとめる。

### 目的

- 電池材料の充放電曲線（Voltage vs. Capacity）に特化したLine抽出モデルの構築
- 事前学習済みモデル（一般的な線グラフ用）と Fine-tuning モデルの比較

---

## 2. データセット

WebPlotDigitizer（WPD）形式のアノテーション付き電池充放電曲線データを使用。

| 項目 | 値 |
|------|-----|
| データソース | WPD_file（tar形式） |
| フィルタ条件 | Y軸: 1.0-6.0V（電圧）, X軸 > 50（容量） |
| 除外 | SID51009 Fig4d-h（部分アノテーション） |
| Train | 62 images / 482 line annotations |
| Val | 19 images / 146 line annotations |
| 画像サイズ | 512 x 512（リサイズ + パディング） |
| カテゴリ | `line`（1クラス） |
| RepeatDataset | 100倍（実効 6,200 samples） |

---

## 3. モデル・学習設定

| 項目 | 値 |
|------|-----|
| ベースモデル | Mask2Former + Swin Transformer Tiny |
| 事前学習チェックポイント | iter_3000.pth（LineFormer公式） |
| Optimizer | AdamW (lr=2e-5, weight_decay=0.05) |
| Backbone lr_mult | 0.1 |
| Batch size | 8 (samples_per_gpu) |
| Workers | 8 (workers_per_gpu) |
| Max iterations | 5,000 |
| LR schedule | Step decay (gamma=0.5 @ iter 2000) |
| Warmup | Linear, 100 iters, ratio=0.01 |
| Evaluation interval | 100 iters |
| Augmentation | RandomFlip (H: 0.3, V: 0.3) |

---

## 4. Training Curves

### 4.1 Training Loss

![Training Loss](images/training_loss.png)

Lossは学習開始時の~27から最終的に~13まで単調に減少した。LR step decay（iter 2000, 4000）の効果が明確に見られる。

### 4.2 Validation segm_mAP

![Validation mAP](images/validation_map.png)

| Iteration | segm_mAP | segm_mAP_50 | 備考 |
|-----------|----------|-------------|------|
| 100 | 0.1260 | 0.5078 | 初期 |
| 500 | 0.1540 | 0.5625 | 急速改善 |
| 700 | 0.1578 | 0.5775 | |
| **1300** | **0.1587** | **0.5710** | **Best mAP** |
| 1400 | 0.1571 | 0.5884 | Best mAP_50 |
| 2000 | 0.1365 | 0.5345 | LR decay後 |
| 3000 | 0.1323 | 0.5178 | |
| 4000 | 0.1199 | 0.4786 | |
| 5000 | 0.1182 | 0.4732 | 最終 |

**Best segm_mAP = 0.1587 @ iter 1300**

mAPはiter 1300をピークに徐々に低下し、過学習（overfitting）の傾向が見られた。62枚という小規模なデータセットでは典型的な現象である。

---

## 5. Valデータセット全体での推論比較

Valデータセット19枚全体に対して、以下の3つのチェックポイントで推論を行い、検出されたライン数とGround Truth（GT）のライン数を比較した。検出の閾値は score > 0.3 とした。

- **Pre-trained (iter_3000)**: Fine-tuning前の事前学習モデル
- **Fine-tuned Best (iter_1300)**: Validation mAPが最大のチェックポイント
- **Fine-tuned (iter_5000)**: 最終チェックポイント

### 5.1 全体サマリー

| モデル | GT合計 | 検出合計 | 過剰検出率 | 検出信頼度平均 |
|--------|--------|---------|-----------|---------------|
| Pre-trained (iter_3000) | 146 | 237 | +62.3% | 0.791 |
| **Fine-tuned Best (iter_1300)** | **146** | **179** | **+22.6%** | **0.872** |
| Fine-tuned (iter_5000) | 146 | 160 | +9.6% | 0.896 |

※「検出信頼度平均」はモデルが各検出に対して出力するconfidence scoreの単純平均であり、GT線とのマッチングに基づくスコアではない。GT線との対応付けを行う元論文準拠の評価（metric6a/6b）はセクション6を参照。

Fine-tuningにより、過剰検出が大幅に削減された。Pre-trainedモデルでは237本（GT比+62.3%）もの過剰検出が発生していたが、Fine-tuned Best (iter_1300) では179本（+22.6%）、Fine-tuned (iter_5000) では160本（+9.6%）まで削減された。

### 5.2 画像別の詳細比較

| 画像 | GT | Pre-trained | Best (iter_1300) | iter_5000 |
|------|-----|-------------|-----------------|-----------|
| SID51027_Fig4b | 6 | 7 | 7 | 8 |
| SID51027_Fig4c | 8 | 5 | 9 | 7 |
| SID51027_Fig4e | 6 | 5 | 5 | 6 |
| SID51027_Fig4h | 6 | 6 | 6 | 6 |
| SID51028_Fig4A | 6 | **14** | 11 | 8 |
| SID51028_Fig4B | 6 | **11** | **14** | 9 |
| SID51029_Fig1a | 12 | 13 | 10 | 9 |
| SID51029_Fig1b | 12 | **17** | 12 | 11 |
| SID51029_Fig3a | 6 | 7 | 8 | 6 |
| SID51029_Fig4a | 12 | **18** | 9 | 9 |
| SID51029_Fig4b | 10 | **17** | 9 | 8 |
| SID51030_Fig3a | 6 | 6 | 6 | 6 |
| SID51030_Fig3b | 4 | **17** | **14** | **17** |
| SID51032_Fig8a | 4 | 4 | 4 | 4 |
| SID51032_Fig8c | 8 | 8 | 5 | 6 |
| SID51033_Fig14b | 6 | **16** | **15** | 9 |
| SID51033_Fig15b | 6 | **31** | 13 | 12 |
| SID51104_2g | 6 | 5 | 5 | 5 |
| SID51104_3a | 16 | **30** | 17 | 14 |

**太字**: GTの2倍以上の過剰検出。

Pre-trainedモデルでは、特にSID51033_Fig15b（GT=6, Det=31）やSID51104_3a（GT=16, Det=30）で大量の誤検出が発生していた。Fine-tuningにより、これらの画像での誤検出が大幅に削減されている。

### 5.3 代表的な画像でのオーバーレイ比較

各画像に対して、検出されたラインマスクを色分けしてオーバーレイした可視化結果を示す。

#### SID51033_Fig15b (GT=6本)

Pre-trainedモデルでは31本も検出しており、テキストや軸目盛りなどの非ライン要素まで誤検出している。Fine-tuned Bestでは13本に改善された。

| Pre-trained (31本検出) | Fine-tuned Best (13本検出) | Fine-tuned iter_5000 (12本検出) |
|:-:|:-:|:-:|
| ![](images/val_pretrained/SID51033_Fig15b.png) | ![](images/val_best_iter1300/SID51033_Fig15b.png) | ![](images/val_iter5000/SID51033_Fig15b.png) |

#### SID51029_Fig4a (GT=12本)

Pre-trainedモデルでは18本を検出したのに対し、Fine-tuned Bestでは9本と過検出が解消された。

| Pre-trained (18本検出) | Fine-tuned Best (9本検出) | Fine-tuned iter_5000 (9本検出) |
|:-:|:-:|:-:|
| ![](images/val_pretrained/SID51029_Fig4a.png) | ![](images/val_best_iter1300/SID51029_Fig4a.png) | ![](images/val_iter5000/SID51029_Fig4a.png) |

#### SID51030_Fig3a (GT=6本) — 良好な検出例

3モデルとも正確に6本を検出。Fine-tunedモデルではスコアが安定して高い。

| Pre-trained (6本, avg=0.797) | Fine-tuned Best (6本, avg=0.944) | Fine-tuned iter_5000 (6本, avg=0.949) |
|:-:|:-:|:-:|
| ![](images/val_pretrained/SID51030_Fig3a.png) | ![](images/val_best_iter1300/SID51030_Fig3a.png) | ![](images/val_iter5000/SID51030_Fig3a.png) |

### 5.4 Fine-tuning後も精度が低かった画像の分析

Fine-tuned Best (iter_1300) においても、GTとの乖離が大きかった画像を以下に示す。

#### SID51030_Fig3b (GT=4, 検出=14, +10 過剰)

30サイクル分の充放電曲線が密集している画像。GTでは充放電のペアを1本とカウントしているが、モデルは密集したラインの断片を個別に検出してしまっている。全モデル（Pre-trained: 17, Best: 14, iter_5000: 17）で大幅な過剰検出が発生しており、ラインが密集するパターンへの根本的な対策が必要。

| Fine-tuned Best (14本検出) |
|:-:|
| ![](images/val_best_iter1300/SID51030_Fig3b.png) |

#### SID51033_Fig14b (GT=6, 検出=15, +9 過剰)

9サイクル分の充放電曲線。急峻な充電カーブと平坦な放電プラトーの組み合わせで、1本のラインが複数のセグメントに分断されて検出される傾向がある。

| Fine-tuned Best (15本検出) |
|:-:|
| ![](images/val_best_iter1300/SID51033_Fig14b.png) |

#### SID51028_Fig4B (GT=6, 検出=14, +8 過剰)

破線（dashed line）や矢印注釈が含まれており、非ライン要素も検出されている。また、図が横方向にクロップされた画像であり、通常とは異なるレイアウトに対する汎化が不十分と考えられる。

| Fine-tuned Best (14本検出) |
|:-:|
| ![](images/val_best_iter1300/SID51028_Fig4B.png) |

#### SID51033_Fig15b (GT=6, 検出=13, +7 過剰)

Pre-trained（31本）からは大幅に改善されたが、依然としてGTの2倍以上を検出。10サイクル分の曲線が密集しており、交差部分でのライン分離が困難なケース。

| Fine-tuned Best (13本検出) |
|:-:|
| ![](images/val_best_iter1300/SID51033_Fig15b.png) |

#### 共通する傾向と原因分析

| 要因 | 該当画像 | 説明 |
|------|---------|------|
| ライン密集 | SID51030_Fig3b, SID51033_Fig14b, SID51033_Fig15b | 多サイクル（10-30本）の曲線が狭い領域に密集し、1本のラインが複数に分断される |
| 破線・注釈 | SID51028_Fig4B | 破線、矢印、寸法注釈などの非ライン要素を誤検出 |
| GTの定義不一致 | SID51030_Fig3b | GTが充放電ペアを1本とカウントしている可能性があり、モデルの検出数と直接比較できない場合がある |

---

## 6. 元論文準拠の評価（metric6a / metric6b）

セクション5の評価はライン検出数と検出信頼度の比較に留まっていたが、GT線と検出線の**対応付け**（マッチング）は行っていなかった。本セクションでは、LineFormer元論文（ICDAR 2023）で使用されているTask 6a / Task 6bメトリクスにより、GT線と検出線の定量的な比較を行う。

### 6.1 評価手法

元論文の評価パイプラインは以下の通り:

1. **推論**: モデルがマスクを検出 → マスクからデータ点(x, y)を抽出（`infer.get_dataseries()`）
2. **GT準備**: WPDアノテーションからGTデータ点を取得し、512x512画像座標に変換
3. **マッチング**: 線形補間 + ハンガリアンアルゴリズム（`scipy.optimize.linear_sum_assignment`）で最適な1対1ペアリング
4. **スコア計算**: 各ペアについて補間誤差のF-score（precision × recall の調和平均）を算出

**Task 6a と Task 6b の違い:**

- **Task 6a**: スコア行列をGT線数で正規化。GT線がどれだけ正確に検出されたかを測る。過剰検出に対するペナルティなし
- **Task 6b**: スコア行列を正方行列にパディングしてから正規化。**過剰検出にペナルティ**がかかるため、実用上はこちらがより重要な指標

### 6.2 全体サマリー

| モデル | Task 6a | Task 6b | GT合計 | 検出合計 |
|--------|---------|---------|--------|---------|
| Pre-trained (iter_3000) | **0.9471** | 0.6835 | 146 | 237 |
| Fine-tuned Best (iter_1300) | 0.9180 | 0.7394 | 146 | 179 |
| **Fine-tuned Final (iter_5000)** | 0.9097 | **0.7836** | **146** | **160** |

**分析:**

- **Task 6a**: Pre-trainedが最高スコア（0.9471）。過剰検出（237本）するため、各GT線に対して非常に近い検出が存在する。ただし余分な誤検出は評価されない
- **Task 6b**: Fine-tuned Final（iter_5000）が最高スコア（0.7836）。検出数がGTに最も近く（160本 vs GT 146本）、過剰検出のペナルティが最小
- Fine-tuningにより、Task 6b（実用指標）が0.6835 → 0.7836へと**14.6%改善**

### 6.3 画像別の詳細スコア

| 画像 | GT | Pre-trained 6a | Pre-trained 6b | Best 6a | Best 6b | iter_5000 6a | iter_5000 6b |
|------|-----|----------------|----------------|---------|---------|--------------|--------------|
| SID51027_Fig4b | 6 | 0.981 | 0.841 | 0.995 | 0.853 | 0.996 | 0.747 |
| SID51027_Fig4c | 8 | 0.619 | 0.619 | 0.916 | 0.814 | 0.870 | 0.870 |
| SID51027_Fig4e | 6 | 0.799 | 0.799 | 0.829 | 0.829 | 0.985 | 0.985 |
| SID51027_Fig4h | 6 | 0.995 | 0.995 | 0.996 | 0.996 | 0.995 | 0.995 |
| SID51028_Fig4A | 6 | 0.993 | 0.426 | 0.994 | 0.542 | 0.991 | 0.744 |
| SID51028_Fig4B | 6 | 0.994 | 0.542 | 0.995 | 0.426 | 0.991 | 0.661 |
| SID51029_Fig1a | 12 | 0.982 | 0.906 | 0.828 | 0.828 | 0.746 | 0.746 |
| SID51029_Fig1b | 12 | 0.995 | 0.702 | 0.929 | 0.929 | 0.900 | 0.900 |
| SID51029_Fig3a | 6 | 0.959 | 0.822 | 0.981 | 0.735 | 0.961 | 0.961 |
| SID51029_Fig4a | 12 | 0.993 | 0.662 | 0.744 | 0.744 | 0.733 | 0.733 |
| SID51029_Fig4b | 10 | 0.975 | 0.573 | 0.886 | 0.886 | 0.782 | 0.782 |
| SID51030_Fig3a | 6 | 0.974 | 0.974 | 0.996 | 0.996 | 0.996 | 0.996 |
| SID51030_Fig3b | 4 | 0.981 | 0.231 | 0.984 | 0.281 | 0.983 | 0.231 |
| SID51032_Fig8a | 4 | 0.995 | 0.995 | 0.996 | 0.996 | 0.996 | 0.996 |
| SID51032_Fig8c | 8 | 0.989 | 0.989 | 0.622 | 0.622 | 0.745 | 0.745 |
| SID51033_Fig14b | 6 | 0.989 | 0.371 | 0.991 | 0.396 | 0.980 | 0.653 |
| SID51033_Fig15b | 6 | 0.980 | 0.190 | 0.986 | 0.455 | 0.981 | 0.490 |
| SID51104_2g | 6 | 0.830 | 0.830 | 0.831 | 0.831 | 0.831 | 0.831 |
| SID51104_3a | 16 | 0.970 | 0.518 | 0.942 | 0.886 | 0.822 | 0.822 |

### 6.4 metric6aの結果に基づく分析

**Task 6bスコアが低い画像（全モデル共通の課題）:**

- **SID51030_Fig3b** (6b ≈ 0.23-0.28): 30サイクルの密集曲線。全モデルで17本前後の過剰検出が発生し、6bスコアが著しく低い。GTが充放電ペアを1本とカウントしている可能性もある
- **SID51033_Fig15b** (6b = 0.19-0.49): 10サイクルの密集曲線。Pre-trainedでは6b=0.19と壊滅的だが、Fine-tuningで0.49まで改善
- **SID51033_Fig14b** (6b = 0.37-0.65): 9サイクルの充放電曲線。iter_5000で0.65まで改善

**Fine-tuningで特に改善した画像:**

- **SID51027_Fig4c**: 6a: 0.619 → 0.870（+40.5%）、6b: 0.619 → 0.870（+40.5%）
- **SID51033_Fig14b**: 6b: 0.371 → 0.653（+76.0%）
- **SID51033_Fig15b**: 6b: 0.190 → 0.490（+157.9%）
- **SID51104_3a**: 6b: 0.518 → 0.822（+58.7%）

---

## 7. 考察

### Fine-tuningの効果

1. **過剰検出の大幅削減**: Pre-trainedモデルはValデータセット全体でGT比+62.3%の過剰検出であったが、Fine-tuned Bestでは+22.6%、最終モデルでは+9.6%まで改善された
2. **検出信頼度の安定化**: 平均confidence scoreが0.791から0.872（Best）/0.896（最終）に向上
3. **Task 6b（実用指標）の改善**: 0.6835 → 0.7836（+14.6%）。GT線とのマッチング精度が向上し、過剰検出ペナルティも減少
4. **ドメイン特化**: 電池充放電曲線に特有のパターン（テキスト注釈、凡例、複数のCレート曲線の密集）に対する誤検出が減少

### 過学習について

- mAPはiter 1300でピーク（0.1587）に達し、その後徐々に低下した
- Lossは最後まで低下し続けており、学習データへの過適合が示唆される
- ただし推論結果（検出数とスコア）はiter_5000でも良好であり、mAPの低下はマスク形状の精度劣化に起因すると推測される
- 62枚のTrainデータでは早期停止（early stopping）が有効である

### 残課題

- **SID51030_Fig3b**: 全モデルでGT=4に対して14-17本を検出しており、特定の画像パターンへの対応に課題がある
- **mAP_75 ≈ 0**: マスクのIoU精度が低い。Line maskの太さ（thickness）の最適化や、post-processingの導入が有効と考えられる
- **検出不足**: 一部の画像（SID51029_Fig1a等）ではGTより少ない検出数となっており、密集ラインの分離精度に改善の余地がある

### 改善の方向性

- **早期停止**: max_iters を 1500-2000 に設定
- **データ拡張の強化**: ColorJitter, RandomRotate 等の追加
- **学習データの増量**: より多くの充放電曲線画像の収集
- **Cosine Annealing**: Step decayよりも滑らかな学習率減衰
- **Line mask thickness の最適化**: mAP_75改善のため

---

## 8. ファイル構成

```
work_dirs/battery_finetune/
  best_segm_mAP_iter_1300.pth   # Best model (推奨)
  iter_5000.pth / latest.pth    # Final model
  iter_4000.pth                 # Intermediate checkpoint
  20260220_180100.log           # Training log

reports/battery_finetune/
  lineformer_battery_finetune_results.md   # 本報告書
  val_results.json                         # Val推論の数値データ
  metric6a_comparison.csv                  # metric6a/6b評価結果（全モデル比較）
  images/
    training_loss.png                      # Loss曲線
    validation_map.png                     # mAP曲線
    val_pretrained/                        # Pre-trained推論結果 (19枚)
    val_best_iter1300/                     # Best model推論結果 (19枚)
    val_iter5000/                          # Final model推論結果 (19枚)
```

## 9. 使用方法

### Valデータセットでの評価（検出数・信頼度）

```bash
python scripts/eval_val_dataset.py \
  --config lineformer_finetune_battery_gpu.py \
  --output-dir reports/battery_finetune
```

### Valデータセットでの評価（metric6a/6b, 元論文準拠）

```bash
# 全3モデルの比較
python scripts/eval_metric6a_battery.py --compare-all

# 単一モデルの評価
python scripts/eval_metric6a_battery.py \
  --config lineformer_swin_t_config.py \
  --checkpoint work_dirs/battery_finetune/iter_5000.pth
```

### 単一画像の推論

```bash
python scripts/run_inference_report.py \
  --image <image_path> \
  --config lineformer_finetune_battery_gpu.py \
  --output-dir reports/output
```

### 学習再開（早期停止版）

```bash
# max_iters を 1500 に変更して再学習
python mmdetection/tools/train.py \
  lineformer_finetune_battery_gpu.py \
  --work-dir work_dirs/battery_finetune_v2 \
  --gpu-id 0 --seed 42
```
