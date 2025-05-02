# DLAV Project - End-to-End Trajectory Planner

**Authors:** Rayan Gauderon & Mouhamad Rawas  
**Course:** Deep Learning for Autonomous Vehicles   
**Date:** May 2, 2025

## Overview — Milestone 1

This repository contains the implementation of an end-to-end trajectory planner for autonomous vehicles as part of the DLAV course project. The goal is to predict future trajectories of a self-driving car based on sensor data. 

For Milestone 1, we've implemented a basic end-to-end model that predicts future trajectories based on:
- Camera RGB image
- Vehicle's motion history

**Note:** While the original requirements mentioned using the driving command as an input, we found that integrating this categorical input did not improve model performance and was therefore not included in the final implementation.

## Model & Training Method

### Architecture Overview

Our model, named **RoadMind**, consists of several key components:

1. **Image Encoder**: A custom CNN that processes camera images to extract visual features, consisting of 4 convolutional blocks followed by an adaptive pooling layer and a fully connected layer.

2. **Motion History Encoder**: A bidirectional GRU network that processes the vehicle's historical trajectory data.

3. **Temporal Attention Mechanism**: Focuses on the most relevant parts of the historical trajectory.

4. **Feature Fusion Module**: Combines visual features and trajectory information using both concatenation and highway connections.

5. **Trajectory Decoder**: Generates the predicted future trajectory points.

The architecture incorporates several advanced techniques:
- Temporal attention mechanism
- Highway connections for better gradient flow
- Careful weight initialization strategies
- Dropout for regularization

### Training Configuration

- **Batch Size**: 32
- **Learning Rate**: 5e-4 with AdamW optimizer
- **Weight Decay**: 1e-4
- **Scheduler**: ReduceLROnPlateau with patience=5, factor=0.5
- **Early Stopping**: Patience of 20 epochs
- **Max Epochs**: 150
- **Data Normalization**: Mean=[0.587, 0.605, 0.590], Std=[0.132, 0.125, 0.163]

### Results

On the validation dataset, our model achieved:
- **ADE (Average Displacement Error)**: 1.7
- **FDE (Final Displacement Error)**: 5.1

This performance places our model well within the expected range for Milestone 1, which required an ADE < 2 for full marks.

## Project Structure

```
project/
├── data/                      # Data directory (will be created)
├── model/                     # Model checkpoints directory
├── outputs/
│   ├── checkpoints/           # Training checkpoints
│   ├── examples/              # Visualization examples
│   └── logs/                  # Training logs and curves
├── src/
│   ├── data_loader.py         # Dataset and data loading utilities
│   ├── evaluate.py            # Evaluation script for inference
│   ├── training_pipeline.py   # Main training script
│   ├── model.py               # Model architecture definition
│   ├── train.py               # Training loop implementation
│   └── utils.py               # Utility functions
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
   torch>=2.0.0
   torchvision>=0.15.0
   numpy>=1.23.5
   matplotlib>=3.7.1
   pandas>=2.0.0
   scikit-learn>=1.2.2
   gdown>=4.7.1
   tqdm>=4.65.0
   ```

## Training

To train the model from scratch:

1. Navigate to the src directory:
   ```bash
   cd src
   ```

2. Run the main training script:
   ```bash
   python training_pipeline.py
   ```

The script will:
- Download the dataset if not already present (need to set download to True in the training_pipeline.py file)
- Create the necessary directories
- Train the model with the configuration defined in main.py
- Save checkpoints and the best model
- Generate training curves and example visualizations

If you want to resume training from a checkpoint, the code automatically detects and loads the latest checkpoint.

## Inference

To run inference and generate submission files:

1. Navigate to the src directory:
   ```bash
   cd src
   ```

2. Run the evaluation script:
   ```bash
   python evaluate.py
   ```

This will:
- Load the best saved model
- Run inference on the test dataset
- Generate a submission CSV file in the submission directory

The submission file format follows the Kaggle competition requirements with columns:
- id
- x_1, y_1, x_2, y_2, ..., x_60, y_60

## Visualization

Training curves and example predictions are automatically saved to the outputs directory.

- **Training Curves**: Located at `outputs/logs/training_curves.png`
- **Example Predictions**: Located at `outputs/examples/prediction_examples.png`

These visualizations show the model's performance and sample predictions, helping to understand how well the model is learning to predict future trajectories.
