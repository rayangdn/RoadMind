import torch
import os
import matplotlib.pyplot as plt
    
def plot_results(train_losses, val_losses, val_ade, val_fde):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Losses
    axs[0].plot(train_losses, label='Training Loss')
    axs[0].plot(val_losses, label='Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training and Validation Losses')
    axs[0].legend()
    axs[0].grid(True)

    # Plot ADE and FDE
    axs[1].plot(val_ade, label='ADE')
    axs[1].plot(val_fde, label='FDE')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Displacement Error')
    axs[1].set_title('Validation ADE and FDE')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig('../outputs/training_metrics.png')
    plt.close()
