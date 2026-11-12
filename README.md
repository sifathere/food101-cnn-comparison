# Food-101 Image Classification: Progressive Deep Learning Model Comparison

This project benchmarks four progressive Convolutional Neural Network (CNN) architectures in **PyTorch** on a 10-class subset of the [ETH Zurich Food-101 dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/). Each model introduces specific architectural and training upgrades to solve the limitations of the previous one.

## Model Progression

1. **Model 1 — Baseline Custom CNN:** 3 convolutional blocks (`Conv2d -> ReLU -> MaxPool2d`) with a flattened dense classifier trained from scratch without regularization or augmentation. Demonstrates severe overfitting on noisy food images.
2. **Model 2 — Regularized Custom CNN + Augmentation:** Adds `BatchNorm2d`, `Dropout`, `AdaptiveAvgPool2d` (cutting parameters by 95%), and on-the-fly training augmentations (`RandomCrop`, `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`).
3. **Model 3 — Transfer Learning (Frozen EfficientNet-B0):** Uses an ImageNet-pretrained `EfficientNet-B0` backbone with frozen convolutional weights and a custom linear classification head.
4. **Model 4 — Fine-Tuned EfficientNet-B0:** Unfreezes the top feature blocks (`features[5:]`) of `EfficientNet-B0` and trains with **Label Smoothing (`0.1`)**, **AdamW**, and a **Cosine Annealing learning rate schedule**.

## Benchmark Results (10-Class Subset, 10 Epochs Each)

| Model | Trainable Params | Final Train Acc (%) | Best Val Top-1 Acc (%) | Best Val Top-5 Acc (%) | Overfitting Gap (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model 1: Baseline CNN** | 8,422,218 | XX.XX% | XX.XX% | XX.XX% | XX.XX% |
| **Model 2: Regularized CNN** | 423,498 | XX.XX% | XX.XX% | XX.XX% | XX.XX% |
| **Model 3: Frozen EfficientNet-B0** | 12,810 | XX.XX% | XX.XX% | XX.XX% | XX.XX% |
| **Model 4: Fine-Tuned EfficientNet-B0** | ~3,500,000 | XX.XX% | XX.XX% | XX.XX% | XX.XX% |

## Dataset Setup
* **Source:** ETHZ Food-101 (automatically downloaded via `torchvision.datasets.Food101`).
* **Split:** 750 training images and 250 test images per class (7,500 train / 2,500 test for the 10-class benchmark).
* **Input Resolution:** `128 x 128` RGB images normalized with ImageNet channel statistics.

## How to Run
1. Open `Food101_CNN_Comparison_Project.ipynb` directly in **Google Colab** using the badge at the top of the notebook.
2. Set the hardware accelerator to **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`).
3. Run all cells sequentially. The dataset downloads directly to Colab's local disk.
