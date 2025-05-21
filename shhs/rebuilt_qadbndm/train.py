#!/usr/bin/env python
"""
QADBNDM Training Script

This script provides a complete pipeline for training and evaluating QADBNDM models
with various quantum configurations on image datasets.
"""

import os
import sys
import json
import argparse
import time
import logging
from pathlib import Path
from datetime import datetime
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from rebuilt_qadbndm.model.qadbndm import QADBNDM
from rebuilt_qadbndm.model.quantum.quantum_sampler import QuantumSampler
from rebuilt_qadbndm.scripts.generate_quantum_configs import load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('qadbndm_train')

def seed_everything(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train QADBNDM model")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--qubit-count", type=int, default=2048, 
                      help="Qubit count for model configuration (ignored if --config is used)")
    parser.add_argument("--epochs", type=int, default=100, 
                      help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, 
                      help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, 
                      help="Learning rate")
    parser.add_argument("--save-dir", type=str, default="outputs",
                      help="Directory to save checkpoints and outputs")
    parser.add_argument("--sample-interval", type=int, default=10,
                      help="Interval (in epochs) to generate and save samples")
    parser.add_argument("--checkpoint-interval", type=int, default=10,
                      help="Interval (in epochs) to save checkpoints")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset", type=str, default="cifar10", 
                      choices=["cifar10", "mnist", "fashion_mnist"],
                      help="Dataset to train on")
    parser.add_argument("--test-mode", action="store_true",
                      help="Run in test mode with 1 epoch and small dataset")
    parser.add_argument("--multi-anneal", action="store_true",
                      help="Use multi-anneal quantum sampling")
    parser.add_argument("--num-samples", type=int, default=16,
                      help="Number of samples to generate during testing")
    parser.add_argument("--diffusion-steps", type=int, default=16,
                      help="Number of diffusion steps for generation")
    return parser.parse_args()

def load_dataset(dataset_name, batch_size, test_mode=False):
    """
    Load dataset for training and testing.
    
    Args:
        dataset_name: Name of dataset to load
        batch_size: Batch size for DataLoader
        test_mode: If True, use a tiny subset for testing
        
    Returns:
        train_loader, test_loader
    """
    # Set up data transforms
    if dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)
        
    elif dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
        
    elif dataset_name == "fashion_mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform)
    
    # For test mode, use small subsets
    if test_mode:
        logger.info("Using test mode with small dataset")
        train_indices = torch.randperm(len(train_dataset))[:100]  # Just 100 training samples
        test_indices = torch.randperm(len(test_dataset))[:20]  # Just 20 test samples
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    
    logger.info(f"Loaded {dataset_name} dataset")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Testing samples: {len(test_dataset)}")
    
    return train_loader, test_loader

def create_model_from_config(config, device, multi_anneal=False):
    """
    Create a QADBNDM model from configuration.
    
    Args:
        config: Model configuration dictionary
        device: Device to place model on
        multi_anneal: Whether to use multi-anneal sampling
        
    Returns:
        Instantiated QADBNDM model
    """
    # Determine input channels based on image shape in config
    if len(config['image_size']) == 2:
        # Default to RGB if only height/width specified
        input_shape = (3, config['image_size'][0], config['image_size'][1])
    else:
        input_shape = tuple(config['image_size'])
    
    # Create model
    logger.info(f"Creating QADBNDM model with shape {input_shape}")
    logger.info(f"Latent dimensions: {config['model_parameters']['latent_dims']}")
    logger.info(f"Scale factors: {config['model_parameters']['scale_factors']}")
    
    model = QADBNDM(
        input_shape=input_shape,
        hidden_channels=config['model_parameters']['hidden_channels'],
        latent_dims=config['model_parameters']['latent_dims'],
        scale_factors=config['model_parameters']['scale_factors'],
        use_attention=config['model_parameters']['use_attention'],
        use_quantum=True,  # Always use quantum for consistency
        binary_temp_schedule=True,  # Use temperature schedule for binarization
        device=device
    )
    
    # Configure quantum sampler
    use_multi_anneal = multi_anneal or config['quantum_parameters'].get('multi_anneal', False)
    
    model.quantum_sampler = QuantumSampler(
        latent_size=model.encoder.get_total_latent_size(),
        max_batch_size=config['model_parameters']['dbns_per_run'],
        multi_anneal=use_multi_anneal,
        anneals_per_run=config['quantum_parameters'].get('anneals_per_run', 5),
        top_k_percent=config['quantum_parameters'].get('top_k_percent', 20.0),
        target_batch_size=config['quantum_parameters'].get('target_batch_size', 4),
        device=device
    )
    
    return model

def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch
        loss: Current loss
        save_path: Path to save checkpoint to
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path}")

def load_checkpoint(model, optimizer, load_path, device):
    """
    Load model checkpoint.
    
    Args:
        model: Model to load weights into
        optimizer: Optimizer to load state into
        load_path: Path to load checkpoint from
        device: Device to load tensors to
        
    Returns:
        epoch, loss
    """
    if not os.path.exists(load_path):
        logger.error(f"Checkpoint {load_path} does not exist")
        return 0, float('inf')
    
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    logger.info(f"Loaded checkpoint from {load_path}")
    logger.info(f"Resuming from epoch {epoch+1}")
    
    return epoch, loss

def sample_and_save(model, epoch, num_samples, output_dir, steps=16):
    """
    Generate and save samples from the model.
    
    Args:
        model: QADBNDM model
        epoch: Current epoch (for filename)
        num_samples: Number of samples to generate
        output_dir: Directory to save samples to
        steps: Number of diffusion steps for generation
    """
    logger.info(f"Generating {num_samples} samples with {steps} diffusion steps")
    
    # Generate samples
    start_time = time.time()
    with torch.no_grad():
        samples = model.generate(batch_size=num_samples, steps=steps)
    generation_time = time.time() - start_time
    
    logger.info(f"Generation took {generation_time:.2f}s ({generation_time/num_samples:.2f}s per sample)")
    
    # Save individual samples
    samples_dir = os.path.join(output_dir, "samples", f"epoch_{epoch}")
    os.makedirs(samples_dir, exist_ok=True)
    
    # Rescale from [-1, 1] to [0, 1]
    samples_normalized = (samples + 1) / 2
    
    # Save as grid
    grid = make_grid(samples_normalized, nrow=int(np.sqrt(num_samples)))
    save_image(grid, os.path.join(output_dir, f"samples_epoch_{epoch}.png"))
    
    # Save individual images
    for i in range(num_samples):
        save_image(samples_normalized[i], os.path.join(samples_dir, f"sample_{i+1}.png"))

def evaluate(model, test_loader, device, num_samples=10):
    """
    Evaluate model on test data.
    
    Args:
        model: QADBNDM model
        test_loader: DataLoader with test data
        device: Device to run evaluation on
        num_samples: Number of test samples to use for reconstruction
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    total_loss = 0
    recon_loss = 0
    
    # Get a batch of test data
    test_batch = next(iter(test_loader))
    test_images = test_batch[0][:num_samples].to(device)
    
    # Perform reconstruction
    with torch.no_grad():
        # Encode and reconstruct
        start_time = time.time()
        latents = model.encode(test_images)
        reconstructions = model.decode(latents)
        recon_time = time.time() - start_time
        
        # Calculate reconstruction loss
        recon_loss = nn.MSELoss()(reconstructions, test_images).item()
        
        # Generate random samples
        sample_start_time = time.time()
        samples = model.generate(batch_size=num_samples, steps=16)
        sample_time = time.time() - sample_start_time
    
    metrics = {
        'recon_loss': recon_loss,
        'recon_time': recon_time / num_samples,
        'generation_time': sample_time / num_samples,
    }
    
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics, test_images, reconstructions, samples

def train_epoch(model, optimizer, train_loader, device, epoch, scheduler=None):
    """
    Train model for one epoch.
    
    Args:
        model: QADBNDM model
        optimizer: Optimizer
        train_loader: DataLoader with training data
        device: Device to train on
        epoch: Current epoch number
        scheduler: Optional learning rate scheduler
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        # Move data to device
        data = data.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        loss = model.training_step(data)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update total loss
        total_loss += loss.item()
        
        # Log progress
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.6f}")
    
    # Update scheduler if provided
    if scheduler is not None:
        scheduler.step()
    
    # Calculate average loss
    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch {epoch+1} complete, Avg. Loss: {avg_loss:.6f}")
    
    return avg_loss

def save_training_results(metrics_history, output_dir):
    """
    Save training metrics and plots.
    
    Args:
        metrics_history: Dictionary of metrics over training
        output_dir: Directory to save results to
    """
    # Save metrics as JSON
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    # If matplotlib is available, create plots
    try:
        import matplotlib.pyplot as plt
        
        # Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(metrics_history['train_loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(output_dir, "training_loss.png"))
        
        # Plot evaluation metrics
        if len(metrics_history['eval_recon_loss']) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(metrics_history['eval_recon_loss'], label='Reconstruction Loss')
            plt.title('Evaluation Metrics')
            plt.xlabel('Evaluation Point')
            plt.ylabel('Value')
            plt.legend()
            plt.savefig(os.path.join(output_dir, "eval_metrics.png"))
    except ImportError:
        logger.warning("Matplotlib not available, skipping plot generation")

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    seed_everything(args.seed)
    
    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        args.save_dir, 
        f"{args.dataset}_{args.qubit_count}qubits_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Save args for reference
    with open(os.path.join(output_dir, "args.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset
    train_loader, test_loader = load_dataset(
        args.dataset, args.batch_size, args.test_mode)
    
    # Get a sample to determine input shape
    sample_batch = next(iter(train_loader))
    sample_shape = sample_batch[0].shape[1:]  # [C, H, W]
    logger.info(f"Input shape: {sample_shape}")
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            configs = json.load(f)
            # Find a config matching our image size
            matching_configs = [c for c in configs if c['image_size'][0] == sample_shape[1]]
            config = matching_configs[0] if matching_configs else configs[0]
    else:
        # Use default config from generate_quantum_configs.py
        config = load_config(None, args.qubit_count, sample_shape[1])
        # Update image shape to match dataset
        config['image_size'] = [sample_shape[1], sample_shape[2]]
        if sample_shape[0] != 3:
            # Handle grayscale by repeating the channel
            logger.info(f"Adjusting config for {sample_shape[0]} channels")
        
    # Save config
    config_path = os.path.join(output_dir, "model_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create model
    model = create_model_from_config(config, device, args.multi_anneal)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    # Initialize tracking variables
    start_epoch = 0
    best_loss = float('inf')
    metrics_history = {
        'train_loss': [],
        'eval_recon_loss': [],
        'eval_generation_time': [],
    }
    
    # Resume from checkpoint if specified
    if args.resume:
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume, device)
    
    # Training loop
    logger.info("Starting training...")
    num_epochs = 1 if args.test_mode else args.epochs
    
    for epoch in range(start_epoch, num_epochs):
        # Train for one epoch
        epoch_loss = train_epoch(model, optimizer, train_loader, device, epoch, scheduler)
        metrics_history['train_loss'].append(epoch_loss)
        
        # Save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, epoch, epoch_loss, checkpoint_path)
            
            # Save as "best" if it's the best so far
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_path = os.path.join(output_dir, "best_model.pt")
                save_checkpoint(model, optimizer, epoch, epoch_loss, best_path)
        
        # Evaluate and generate samples
        if (epoch + 1) % args.sample_interval == 0 or epoch == num_epochs - 1:
            # Evaluate on test set
            eval_metrics, test_images, reconstructions, samples = evaluate(
                model, test_loader, device, args.num_samples)
            
            # Update metrics history
            metrics_history['eval_recon_loss'].append(eval_metrics['recon_loss'])
            metrics_history['eval_generation_time'].append(eval_metrics['generation_time'])
            
            # Save samples
            samples_dir = os.path.join(output_dir, "samples")
            os.makedirs(samples_dir, exist_ok=True)
            
            # Save reconstructions
            recon_grid = torch.cat([test_images[:8], reconstructions[:8]], dim=0)
            recon_grid = (recon_grid + 1) / 2  # [-1, 1] -> [0, 1]
            save_image(make_grid(recon_grid, nrow=8),
                      os.path.join(samples_dir, f"reconstruction_epoch_{epoch+1}.png"))
            
            # Save generated samples
            samples_normalized = (samples + 1) / 2  # [-1, 1] -> [0, 1]
            save_image(make_grid(samples_normalized, nrow=4),
                      os.path.join(samples_dir, f"samples_epoch_{epoch+1}.png"))
            
            # Generate individual samples if in test mode (final evaluation)
            if epoch == num_epochs - 1 or args.test_mode:
                sample_and_save(model, epoch + 1, args.num_samples, output_dir, args.diffusion_steps)
        
        # Save metrics after each epoch
        save_training_results(metrics_history, output_dir)
    
    logger.info("Training complete!")
    
    # Final evaluation with different configuration options if in test mode
    if args.test_mode:
        logger.info("Running final evaluation with different configurations...")
        
        # Test with different qubit counts if in test mode
        for qubit_count in [2048, 5760, 8192]:
            if qubit_count == args.qubit_count:
                continue  # Skip the one we already used
                
            logger.info(f"Testing with {qubit_count} qubit configuration")
            test_config = load_config(None, qubit_count, sample_shape[1])
            test_config['image_size'] = [sample_shape[1], sample_shape[2]]
            
            # Create model with this config
            test_model = create_model_from_config(test_config, device, args.multi_anneal)
            
            # Generate samples
            test_dir = os.path.join(output_dir, f"test_{qubit_count}qubits")
            os.makedirs(test_dir, exist_ok=True)
            
            # Save the test config
            with open(os.path.join(test_dir, "config.json"), 'w') as f:
                json.dump(test_config, f, indent=2)
            
            # Generate samples with this config
            sample_and_save(test_model, 0, args.num_samples, test_dir, args.diffusion_steps)

if __name__ == "__main__":
    main() 