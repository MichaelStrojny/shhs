import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from rebuilt_qadbndm.model.qadbndm import QADBNDM

def parse_args():
    parser = argparse.ArgumentParser(description="Train QADBNDM model")
    
    # Basic parameters
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist", "cifar10", "celeba"],
                      help="Dataset to use")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="results", help="Directory to save results")
    
    # Model parameters
    parser.add_argument("--hidden_channels", type=int, default=128, help="Hidden channels")
    parser.add_argument("--latent_dims", type=str, default="32,16,8", help="Comma-separated latent dimensions for each level")
    parser.add_argument("--scale_factors", type=str, default="8,4,2", help="Comma-separated scale factors for each level")
    parser.add_argument("--no_attention", action="store_true", help="Disable attention in encoder/decoder")
    parser.add_argument("--use_quantum", action="store_true", help="Enable quantum sampling")
    parser.add_argument("--no_temp_schedule", action="store_true", help="Disable temperature scheduling")
    
    # Training parameters
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--binary_weight", type=float, default=0.05, help="Binary entropy loss weight")
    parser.add_argument("--kl_weight", type=float, default=0.1, help="KL divergence loss weight")
    parser.add_argument("--save_interval", type=int, default=10, help="Epoch interval for saving model")
    parser.add_argument("--device", type=str, default=None, help="Device to use (default: auto)")
    parser.add_argument("--max_batches_per_epoch", type=int, default=None,
                          help="Maximum number of batches to process per epoch (for quick testing)")
    
    return parser.parse_args()

def setup_dataset(args):
    """Set up dataset and data loaders"""
    if args.dataset == "mnist" or args.dataset == "fashion_mnist":
        # For MNIST-type datasets
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        if args.dataset == "mnist":
            train_dataset = torchvision.datasets.MNIST(
                root="data", train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.MNIST(
                root="data", train=False, download=True, transform=transform
            )
        else:  # fashion_mnist
            train_dataset = torchvision.datasets.FashionMNIST(
                root="data", train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.FashionMNIST(
                root="data", train=False, download=True, transform=transform
            )
        
        input_shape = (1, 28, 28)
    
    elif args.dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root="data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="data", train=False, download=True, transform=transform
        )
        
        input_shape = (3, 32, 32)
    
    elif args.dataset == "celeba":
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        
        train_dataset = torchvision.datasets.CelebA(
            root="data", split="train", download=True, transform=transform
        )
        test_dataset = torchvision.datasets.CelebA(
            root="data", split="test", download=True, transform=transform
        )
        
        input_shape = (3, 64, 64)
    
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Create data loaders
    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    return train_loader, test_loader, input_shape

def create_model(args, input_shape):
    """Create QADBNDM model instance"""
    # Parse latent dimensions and scale factors
    latent_dims = [int(dim) for dim in args.latent_dims.split(",")]
    scale_factors = [int(factor) for factor in args.scale_factors.split(",")]
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = QADBNDM(
        input_shape=input_shape,
        hidden_channels=args.hidden_channels,
        latent_dims=latent_dims,
        scale_factors=scale_factors,
        use_attention=not args.no_attention,
        use_quantum=args.use_quantum,
        binary_temp_schedule=not args.no_temp_schedule,
        device=device
    )
    
    # Set loss weights
    model.set_binary_weight(args.binary_weight)
    model.set_kl_weight(args.kl_weight)
    
    return model.to(device)

def save_reconstructions(model, data_loader, device, save_path, num_samples=10):
    """Save visualization of reconstructions"""
    model.eval()
    
    # Get batch of samples
    data_iter = iter(data_loader)
    samples, _ = next(data_iter)
    samples = samples[:num_samples].to(device)
    
    # Reconstruction with and without denoising
    with torch.no_grad():
        outputs_with_denoise = model(samples, denoise_noise_level=0.5)
        outputs_without_denoise = model(samples, denoise_noise_level=None)
        
        recon_with_denoise = outputs_with_denoise['recon']
        recon_without_denoise = outputs_without_denoise['recon']
    
    # Generate random samples
    with torch.no_grad():
        random_samples = model.generate(batch_size=num_samples)
    
    # Create figure
    fig, axs = plt.subplots(4, num_samples, figsize=(num_samples * 2, 8))
    
    # Plot original samples
    for i in range(num_samples):
        img = samples[i].cpu().numpy()
        if img.shape[0] == 1:  # Grayscale
            axs[0, i].imshow(img[0], cmap='gray')
        else:  # RGB
            axs[0, i].imshow(np.transpose(img, (1, 2, 0)))
        axs[0, i].axis('off')
        
        if i == 0:
            axs[0, i].set_title("Original", fontsize=10)
    
    # Plot reconstructions without denoising
    for i in range(num_samples):
        img = recon_without_denoise[i].cpu().numpy()
        if img.shape[0] == 1:  # Grayscale
            axs[1, i].imshow(img[0], cmap='gray')
        else:  # RGB
            axs[1, i].imshow(np.transpose(img, (1, 2, 0)))
        axs[1, i].axis('off')
        
        if i == 0:
            axs[1, i].set_title("Recon (No Denoise)", fontsize=10)
    
    # Plot reconstructions with denoising
    for i in range(num_samples):
        img = recon_with_denoise[i].cpu().numpy()
        if img.shape[0] == 1:  # Grayscale
            axs[2, i].imshow(img[0], cmap='gray')
        else:  # RGB
            axs[2, i].imshow(np.transpose(img, (1, 2, 0)))
        axs[2, i].axis('off')
        
        if i == 0:
            axs[2, i].set_title("Recon (With Denoise)", fontsize=10)
    
    # Plot random samples
    for i in range(num_samples):
        img = random_samples[i].cpu().numpy()
        if img.shape[0] == 1:  # Grayscale
            axs[3, i].imshow(img[0], cmap='gray')
        else:  # RGB
            axs[3, i].imshow(np.transpose(img, (1, 2, 0)))
        axs[3, i].axis('off')
        
        if i == 0:
            axs[3, i].set_title("Generated Samples", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_latents(model, data_loader, device, save_path):
    """Visualize binary latent space"""
    model.eval()
    
    # Get batch of samples
    data_iter = iter(data_loader)
    samples, labels = next(data_iter)
    samples = samples[:100].to(device)  # Use 100 samples
    labels = labels[:100].cpu().numpy()
    
    # Get binary latents
    with torch.no_grad():
        outputs = model(samples, denoise_noise_level=None)
        binary_latents = outputs['encoded_latents']
    
    # For visualization, we'll use the first level latent (coarsest)
    latent = binary_latents[-1]  # Last in the list is coarsest
    
    # Flatten latent to 2D
    batch_size = latent.shape[0]
    flat_latent = latent.view(batch_size, -1).cpu().numpy()
    
    # Use PCA to reduce to 2D for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(flat_latent)
    
    # Create scatter plot colored by class
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', alpha=0.8)
    plt.colorbar(scatter, label='Class')
    plt.title('Binary Latent Space Visualization (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_epoch(model, train_loader, optimizer, device, epoch, max_epochs, args):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_recon_loss = 0.0
    running_binary_loss = 0.0
    running_denoise_loss = 0.0
    processed_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
    if args.max_batches_per_epoch is not None:
        print(f"  >>> Limiting to {args.max_batches_per_epoch} batches for this epoch.")

    for batch_idx, (data, _) in enumerate(pbar):
        if args.max_batches_per_epoch is not None and batch_idx >= args.max_batches_per_epoch:
            print(f"  >>> Reached max_batches_per_epoch ({args.max_batches_per_epoch}), stopping epoch early.")
            break
        data = data.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data, denoise_noise_level=0.5) # Using 0.5 as default noise level
        reconstruction = outputs['recon']
        binary_latents = outputs['encoded_latents']
        denoised_latents = outputs.get('denoised_latents')
        
        # Compute loss
        loss_dict = model.compute_loss(data, reconstruction, binary_latents, denoised_latents)
        total_loss = loss_dict['total']
        
        # Backward pass and optimize
        total_loss.backward()
        optimizer.step()
        
        # Update running losses
        running_loss += total_loss.item()
        running_recon_loss += loss_dict['reconstruction'].item()
        running_binary_loss += loss_dict['binary'].item()
        if loss_dict['denoising'] is not None:
            running_denoise_loss += loss_dict['denoising'].item()
        processed_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / processed_batches,
            'recon': running_recon_loss / processed_batches,
        })
    
    # Update temperature schedule
    model.update_temperature(epoch, max_epochs)
    
    # Calculate epoch losses
    epoch_loss = running_loss / processed_batches if processed_batches > 0 else 0.0
    epoch_recon_loss = running_recon_loss / processed_batches if processed_batches > 0 else 0.0
    epoch_binary_loss = running_binary_loss / processed_batches if processed_batches > 0 else 0.0
    epoch_denoise_loss = running_denoise_loss / processed_batches if processed_batches > 0 else 0.0
    
    return {
        'total': epoch_loss,
        'reconstruction': epoch_recon_loss,
        'binary': epoch_binary_loss,
        'denoising': epoch_denoise_loss
    }

def evaluate(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    running_loss = 0.0
    running_recon_loss = 0.0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            
            # Forward pass
            outputs = model(data, denoise_noise_level=0.5) # Using 0.5 as default noise level for eval
            reconstruction = outputs['recon']
            binary_latents = outputs['encoded_latents']
            denoised_latents = outputs.get('denoised_latents')
            
            # Compute loss
            loss_dict = model.compute_loss(data, reconstruction, binary_latents, denoised_latents)
            
            # Update running losses
            running_loss += loss_dict['total'].item()
            running_recon_loss += loss_dict['reconstruction'].item()
    
    # Calculate average losses
    num_batches = len(test_loader)
    avg_loss = running_loss / num_batches
    avg_recon_loss = running_recon_loss / num_batches
    
    return avg_loss, avg_recon_loss

def plot_loss_curves(train_losses, val_losses, save_path):
    """Plot and save training and validation loss curves"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup dataset
    train_loader, test_loader, input_shape = setup_dataset(args)
    
    # Create model
    model = create_model(args, input_shape)
    device = next(model.parameters()).device
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Print model and training info
    print(f"Training QADBNDM on {args.dataset} dataset")
    print(f"Input shape: {input_shape}")
    print(f"Device: {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training loop
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Train one epoch
        train_loss_dict = train_epoch(model, train_loader, optimizer, device, epoch, args.epochs, args)
        train_loss = train_loss_dict['total']
        train_losses.append(train_loss)
        
        # Evaluate
        val_loss, val_recon_loss = evaluate(model, test_loader, device)
        val_losses.append(val_loss)
        
        # Print epoch stats
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.6f}, "
              f"Train Recon: {train_loss_dict['reconstruction']:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"Val Recon: {val_recon_loss:.6f}")
        
        # Save visualizations and model
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            # Save reconstructions
            save_reconstructions(
                model, test_loader, device,
                os.path.join(args.save_dir, f"recon_epoch_{epoch+1}.png")
            )
            
            # Visualize latent space
            visualize_latents(
                model, test_loader, device,
                os.path.join(args.save_dir, f"latent_epoch_{epoch+1}.png")
            )
            
            # Save model checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pt"))
    
    # Training complete
    training_time = time.time() - start_time
    print(f"Training completed in {training_time/60:.2f} minutes")
    
    # Plot and save loss curves
    plot_loss_curves(train_losses, val_losses, os.path.join(args.save_dir, "loss_curves.png"))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.save_dir, "final_model.pt"))
    
    # Generate samples
    with torch.no_grad():
        samples = model.generate(batch_size=16)
    
    # Save generated samples
    plt.figure(figsize=(12, 8))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        img = samples[i].cpu().numpy()
        if img.shape[0] == 1:  # Grayscale
            plt.imshow(img[0], cmap='gray')
        else:  # RGB
            plt.imshow(np.transpose(img, (1, 2, 0)))
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "final_samples.png"))

if __name__ == "__main__":
    main() 