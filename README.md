# Cell Microscopy Segmentation: Reproducibility Guide

This repository contains the code and experiments for our project on supervised and unsupervised segmentation of cell microscopy images using deep learning.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Reproducing Results](#reproducing-results)
  - [1. Augmentation Forward Selection](#1-augmentation-forward-selection)
  - [2. Augmentation Amount Experiment](#2-augmentation-amount-experiment)
  - [3. Loss Function Evaluation](#3-loss-function-evaluation)
  - [4. Model Configuration & Final Model Selection](#4-model-configuration--final-model-selection)
- [Results](#results)

---

## Project Overview

This project investigates deep learning methods for cell segmentation in microscopy images. We evaluate different data augmentation strategies, loss functions, and model configurations to optimize segmentation performance.

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies (install with `pip install -r requirements.txt`)
- GPU recommended for training

## Data Preparation

1. Place your labeled images and masks in the following structure:
    ```
    data/manual_labels/
        Labelled_images/
        GT_masks/
    ```
2. Update any data paths in the config dictionaries in the experiment scripts if your data is elsewhere.

## Reproducing Results

All main experiments can be launched from [`main.py`](main.py). Alternatively, you can run each experiment script directly.

### 1. Augmentation Forward Selection

This experiment discovers the best combination of augmentations for training.

```sh
python main.py --experiment augmentation_selection
```

or directly:
```
python forward_selection_integration.py
```

### 2. Augmentation Amount Experiment

This experiment tests how many augmented samples per image are optimal.

```sh
python main.py --experiment augmentation_amount
```
or directly:
```sh
python augmentation_amount_experiment.py
```

### 3. Loss Function Evaluation

This experiment compares different loss functions for segmentation.

```sh
python main.py --experiment loss_function
```
or directly:
```sh
python loss_function_cross_validation.py
```

### 4. Model Configuration & Final Model Selection

This experiment runs a grid search over model hyperparameters and selects the final model.

```sh
python main.py --experiment model_configuration
```
or directly:
```sh
python model_configuration_experiment.py
```

## Results

- All results, plots, and logs are saved in the `experiments/` directory.
- Each experiment script saves its results and summary in a timestamped subfolder.
