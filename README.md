# DLAV Project - End-to-End Trajectory Planner

**Authors:** Rayan Gauderon & Mouhamad Rawas  
**Course:** Deep Learning for Autonomous Vehicles   
**Date:** May 16, 2025

## Overview — Milestone 2

This repository contains the implementation of an end-to-end trajectory planner for autonomous vehicles as part of the DLAV course project. The goal is to predict future trajectories of a self-driving car based on sensor data.

For Milestone 2, we've enhanced our model with perception-aware planning through auxiliary tasks:
- Camera RGB image
- Vehicle's motion history
- **New: Depth estimation as auxiliary task**
- **New: Semantic segmentation as auxiliary task**

## Model & Training Method

### Architecture Overview

Our enhanced model, **RoadMind**, has been upgraded with additional components:

1. **Image Encoder**: We now use a pretrained EfficientNet-B0 backbone to extract visual features, which provides a stronger foundation for our perception tasks.

2. **Trajectory Encoder**: A GRU network (now with 2 layers) that processes the vehicle's historical trajectory data, including optional dynamics features.

3. **Feature Fusion Module**: Combines visual features and trajectory information with enhanced connectivity.

4. **Trajectory Decoder**: Generates the predicted future trajectory points.

5. **New: Depth Decoder**: An auxiliary decoder that predicts depth information from image features, helping the model understand the 3D structure of the scene.

6. **New: Semantic Decoder**: An auxiliary decoder that performs semantic segmentation with 15 classes, providing the model with an understanding of road elements.

The architecture incorporates several advanced techniques:
- Multi-task learning with weighted loss functions
- Auxiliary tasks for improved feature learning
- **New: Dynamics features** (optional) including velocity and acceleration
- Dropout for regularization (optimized rate of 0.4)

### Training Configuration

- **Batch Size**: 64
- **Learning Rate**: 8e-4 with Adam optimizer
- **Weight Decay**: 5.4e-5
- **Scheduler**: ReduceLROnPlateau with patience=5, factor=0.7
- **Hyperparameters**: Optimized with Optuna (10 trials)
- **Max Epochs**: 100
- **Auxiliary task weights**:
  - Depth weight: 10
  - Semantic weight: 0.2

### Results

On the validation dataset, our enhanced model achieved:
- **ADE (Average Displacement Error)**: 1.6
- **FDE (Final Displacement Error)**: 4.6

## Project Structure

```
project/
├── data/
|   └── data_loader.py         # Dataset and data loading utilities         
├── checkpoints/               # Model checkpoints directory
│   └── roadmind/              # RoadMind model checkpoints
├── logs/                      # Training logs and curves
│   └── roadmind/              # RoadMind training logs
├── get_hype.py                # Hyperparameter optimization with Optuna
├── infer.py                   # Evaluation script for inference
├── model.py                   # Model architecture definition
├── train.py                   # Training loop implementation
├── utils.py                   # Utility functions
├── submission/                # Submission files
└── requirements.txt           # Python dependencies
```

## Setup

1. Create a new conda environment with Python 3.11:
   ```bash
   conda create -n dlav_env python=3.11
   conda activate dlav_env
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure to have the following dependencies:
   ```
   torch>=1.10.0
   torchvision>=0.11.0
   pytorch-lightning>=1.5.0
   matplotlib>=3.5.0
   numpy>=1.20.0
   pandas>=1.3.0
   gdown>=4.4.0
   optuna>=2.10.0
   pillow>=8.3.0
   tensorboard>=2.7.0
   ```

## Training

To train the model from scratch:

1. Run the training script:
   ```bash
   python train.py
   ```

The script will:
- Download the dataset if not already present
- Create the necessary directories
- Train the model with the configuration defined in train.py
- Save checkpoints and the best model
- Generate training curves and example visualizations

For hyperparameter optimization:
```bash
python get_hype.py
```

## Inference

To run inference and generate submission files:

```bash
python infer.py
```

This will:
- Load the best saved model
- Run inference on the test dataset
- Generate a submission CSV file in the submission directory

The submission file format follows the Kaggle competition requirements with columns:
- id
- x_1, y_1, x_2, y_2, ..., x_60, y_60

## Visualization

Training curves and example predictions are automatically saved during training:

- **Training Curves**: Located in the `logs/roadmind/` directory
  - Visualize with TensorBoard: `tensorboard --logdir=./logs
- **Example Predictions**: Visualizations include:
  - Camera view
  - Trajectory prediction (history, ground truth, and prediction)
  - Depth map (ground truth and prediction)
  - Semantic segmentation (ground truth and prediction)

These visualizations show the model's performance across all tasks, providing insight into how the auxiliary tasks contribute to improved trajectory prediction. TensorBoard allows you to monitor metrics like loss, ADE, FDE, and learning rate in real-time during training.
