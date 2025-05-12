import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from data.data_loader import NuplanDataLoader, AugmentedNuPlanDataset, get_data_paths, visualize_samples
from model import LightningRoadMind
from utils import plot_examples


def train(num_epochs=50, lr=1e-4, weight_decay=1e-5, scheduler_factor=0.1, scheduler_patience=5,
          precision='high', hidden_dim=128, image_embed_dim=256, num_layers_gru=1, dropout_rate=0.3, 
          include_heading=False, include_dynamics=True,
          batch_size=64, logger_name='roadmind'):

    
    # Set seed for reproducibility
    pl.seed_everything(42)
    
    # Set precision for float32 matmul
    torch.set_float32_matmul_precision(precision)
    
    # Data directories
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Download and extract dataset
    # data_loader = NuplanDataLoader(data_dir=data_dir)  # Uncomment to download the dataset
    
    # Create datasets and data loaders
    data_paths = get_data_paths(data_dir)
    train_dataset = AugmentedNuPlanDataset(data_paths['train'], test=False, include_dynamics=include_dynamics, augment_prob=0.5)
    val_dataset = AugmentedNuPlanDataset(data_paths['val'], test=False, include_dynamics=include_dynamics, augment_prob=0.0) # No augmentation for validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    
    # Visualize some samples
    # visualize_samples(train_dataset, num_samples=4) 
    # visualize_samples(val_dataset, num_samples=4)
    
    model = LightningRoadMind(
        hidden_dim=hidden_dim,
        image_embed_dim=image_embed_dim,
        num_layers_gru=num_layers_gru,
        dropout_rate=dropout_rate,
        include_heading=include_heading,
        include_dynamics=include_dynamics,
        lr=lr,
        weight_decay=weight_decay,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience
    )
    
    # Set up TensorBoard logger and learning rate monitor
    logger = TensorBoardLogger(save_dir='logs', name=logger_name)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # create a directory for the checkpoints
    version = logger.version
    filename = f"roadmind_{version}"
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints'+ f'/{logger_name}/',
        filename= filename + '_{epoch:02d}_{val_ade:.2f}',
        monitor='val_ade',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )

    # Create trainer 
    trainer = pl.Trainer(
        max_epochs=num_epochs, 
        logger=logger,
        accelerator='auto',
        devices='auto',
        precision='16-mixed',  
        log_every_n_steps=5, 
        gradient_clip_val=5.0,
        enable_checkpointing=True,  
        callbacks=[checkpoint_callback, lr_monitor],
    )

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)
    best_model_path = checkpoint_callback.best_model_path
    best_model_val_ade = checkpoint_callback.best_model_score
    print(f"Best model val_ade: {best_model_val_ade} Located at path: {best_model_path}")
    
    # Load the best model
    model = LightningRoadMind.load_from_checkpoint(best_model_path)

    # Test the model on the validation set again to be sure
    model.eval()
    trainer.validate(model, val_dataloader)
    
    # Create output directory for visualizations
    save_dir = os.path.join('./logs' + f'/{logger_name}/', f'version_{version}/examples')
    os.makedirs(save_dir, exist_ok=True)
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Plot examples
    plot_examples(
        model=model, 
        data_loader=val_dataloader, 
        device=device,
        save_dir=save_dir, 
        num_samples=4, 
        testing=False
    )
    
    return model, best_model_val_ade

def main():
    
    # <------ Hyperparameters ---->
    
    # Trainer
    num_epochs = 100
    
    # Optimizer
    lr = 0.00018518676207528317
    weight_decay = 2.6551679570632748e-06
    scheduler_factor = 0.7212519748210666
    scheduler_patience = 5
    precision = 'high'
    
    # Model
    hidden_dim = 128
    image_embed_dim = 256
    num_layers_gru = 1
    dropout_rate = 0.3
    include_heading = False
    include_dynamics = True
    
    # Data
    batch_size = 64
    
    model, best_model_score = train(
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        scheduler_factor=scheduler_factor,
        scheduler_patience=scheduler_patience,
        precision=precision,
        hidden_dim=hidden_dim,
        image_embed_dim=image_embed_dim,
        num_layers_gru=num_layers_gru,
        dropout_rate=dropout_rate,
        include_heading=include_heading,
        include_dynamics=include_dynamics,
        batch_size=batch_size,
        logger_name='roadmind'
    )
    
    print(f"Best model ADE score: {best_model_score}")
        
if __name__ == "__main__":
    main()
    
    
    