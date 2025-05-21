#!/usr/bin/env python
"""
QADBNDM Testing Script

This script evaluates trained QADBNDM models and generates samples
with various configurations, including multi-anneal settings.
"""

import os
import sys
import json
import argparse
import time
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

from rebuilt_qadbndm.model.qadbndm import QADBNDM
from rebuilt_qadbndm.model.quantum.quantum_sampler import QuantumSampler
from rebuilt_qadbndm.scripts.generate_quantum_configs import load_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('qadbndm_test')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test QADBNDM model")
    parser.add_argument("--checkpoint", type=str, required=True, 
                      help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Path to configuration file (optional)")
    parser.add_argument("--qubit-counts", nargs="+", type=int, default=[2048, 5760, 8192],
                      help="Qubit counts to test with (ignored if --config is provided)")
    parser.add_argument("--batch-size", type=int, default=16, 
                      help="Batch size for sampling")
    parser.add_argument("--output-dir", type=str, default="test_results",
                      help="Directory to save outputs")
    parser.add_argument("--steps", type=int, default=16,
                      help="Number of diffusion steps for generation")
    parser.add_argument("--compare-multi-anneal", action="store_true",
                      help="Compare standard vs multi-anneal sampling")
    parser.add_argument("--dataset", type=str, default="cifar10", 
                      choices=["cifar10", "mnist", "fashion_mnist"],
                      help="Dataset to test on")
    parser.add_argument("--anneals-per-run", type=int, default=20,
                      help="Number of anneals per run for multi-anneal mode")
    parser.add_argument("--top-k-percent", type=float, default=20.0,
                      help="Percentage of top solutions for multi-anneal mode")
    return parser.parse_args()

def load_test_dataset(dataset_name, batch_size):
    """
    Load test dataset.
    
    Args:
        dataset_name: Name of dataset to load
        batch_size: Batch size for DataLoader
        
    Returns:
        test_loader, sample_shape
    """
    # Set up data transforms
    if dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)
        
    elif dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
        
    elif dataset_name == "fashion_mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        test_dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=transform)
    
    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # Get sample shape
    sample_batch = next(iter(test_loader))
    sample_shape = sample_batch[0].shape[1:]  # [C, H, W]
    
    logger.info(f"Loaded {dataset_name} test dataset")
    logger.info(f"Test samples: {len(test_dataset)}")
    logger.info(f"Sample shape: {sample_shape}")
    
    return test_loader, sample_shape

def create_model_from_checkpoint(checkpoint_path, config, device):
    """
    Create a model from checkpoint and configuration.
    
    Args:
        checkpoint_path: Path to checkpoint file
        config: Model configuration
        device: Device to place model on
        
    Returns:
        Instantiated QADBNDM model
    """
    # Determine input shape
    if len(config['image_size']) == 2:
        # Default to RGB if only height/width given
        input_shape = (3, config['image_size'][0], config['image_size'][1])
    else:
        input_shape = tuple(config['image_size'])
    
    # Create model
    model = QADBNDM(
        input_shape=input_shape,
        hidden_channels=config['model_parameters']['hidden_channels'],
        latent_dims=config['model_parameters']['latent_dims'],
        scale_factors=config['model_parameters']['scale_factors'],
        use_attention=config['model_parameters']['use_attention'],
        use_quantum=True,
        binary_temp_schedule=True,
        device=device
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Loaded model from checkpoint {checkpoint_path}")
    logger.info(f"Model was trained for {checkpoint['epoch']} epochs")
    
    return model

def configure_quantum_sampler(model, config, multi_anneal=False, anneals_per_run=20, top_k_percent=20.0):
    """
    Configure the quantum sampler for the model.
    
    Args:
        model: QADBNDM model
        config: Model configuration
        multi_anneal: Whether to use multi-anneal sampling
        anneals_per_run: Number of anneals per run
        top_k_percent: Percentage of top solutions to select
        
    Returns:
        Model with configured quantum sampler
    """
    # Check if we should force multi-anneal mode
    use_multi_anneal = multi_anneal or config['quantum_parameters'].get('multi_anneal', False)
    
    model.quantum_sampler = QuantumSampler(
        latent_size=model.encoder.get_total_latent_size(),
        max_batch_size=config['model_parameters']['dbns_per_run'],
        multi_anneal=use_multi_anneal,
        anneals_per_run=anneals_per_run,
        top_k_percent=top_k_percent,
        target_batch_size=config['quantum_parameters'].get('target_batch_size', 16),
        device=model.device
    )
    
    logger.info(f"Configured quantum sampler with multi_anneal={use_multi_anneal}")
    if use_multi_anneal:
        logger.info(f"Anneals per run: {anneals_per_run}, Top-k percent: {top_k_percent}%")
    
    return model

def evaluate_model(model, test_loader, device, batch_size, steps=16):
    """
    Evaluate model on test data.
    
    Args:
        model: QADBNDM model
        test_loader: DataLoader with test data
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        steps: Number of diffusion steps for generation
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Get a batch of test data
    test_batch = next(iter(test_loader))
    test_images = test_batch[0][:batch_size].to(device)
    
    # Perform reconstruction
    with torch.no_grad():
        # Encode and reconstruct
        start_time = time.time()
        latents = model.encode(test_images)
        reconstructions = model.decode(latents)
        recon_time = time.time() - start_time
        
        # Calculate reconstruction loss
        recon_loss = F.mse_loss(reconstructions, test_images).item()
        
        # Calculate PSNR
        mse = F.mse_loss(reconstructions, test_images, reduction='none').mean(dim=[1,2,3])
        psnr = 10 * torch.log10(4.0 / mse)  # 4.0 is the range (-1 to 1) squared
        avg_psnr = psnr.mean().item()
        
        # Generate random samples
        sample_start_time = time.time()
        samples = model.generate(batch_size=batch_size, steps=steps)
        sample_time = time.time() - sample_start_time
        
        # Calculate sample diversity (standard deviation across samples)
        sample_std = samples.std(dim=[0, 2, 3]).mean().item()
    
    metrics = {
        'recon_loss': recon_loss,
        'psnr': avg_psnr,
        'recon_time': recon_time / batch_size,
        'generation_time': sample_time / batch_size,
        'sample_diversity': sample_std
    }
    
    logger.info(f"Evaluation metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    return metrics, test_images, reconstructions, samples

def compare_sampling_methods(model, config, device, output_dir, batch_size=16, steps=16, 
                           anneals_per_run=20, top_k_percent=20.0):
    """
    Compare standard sampling with multi-anneal sampling.
    
    Args:
        model: QADBNDM model
        config: Model configuration
        device: Device to run on
        output_dir: Output directory
        batch_size: Number of samples to generate
        steps: Number of diffusion steps
        anneals_per_run: Number of anneals per run for multi-anneal
        top_k_percent: Percentage of top solutions to select
        
    Returns:
        Standard samples, multi-anneal samples, metrics dictionary
    """
    model.eval()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Comparing standard vs multi-anneal sampling")
    
    # First with standard sampling
    logger.info("Testing with standard quantum sampling...")
    configure_quantum_sampler(model, config, multi_anneal=False)
    
    # Generate samples
    start_time = time.time()
    with torch.no_grad():
        samples_standard = model.generate(batch_size=batch_size, steps=steps)
    standard_time = time.time() - start_time
    logger.info(f"Standard sampling time: {standard_time:.2f}s")
    
    # Calculate diversity
    std_standard = samples_standard.std(dim=[0, 2, 3]).mean().item()
    
    # Now with multi-anneal
    logger.info("Testing with multi-anneal quantum sampling...")
    configure_quantum_sampler(
        model, config, multi_anneal=True, 
        anneals_per_run=anneals_per_run, 
        top_k_percent=top_k_percent
    )
    
    # Generate samples
    start_time = time.time()
    with torch.no_grad():
        samples_multi = model.generate(batch_size=batch_size, steps=steps)
    multi_time = time.time() - start_time
    logger.info(f"Multi-anneal sampling time: {multi_time:.2f}s")
    
    # Calculate diversity
    std_multi = samples_multi.std(dim=[0, 2, 3]).mean().item()
    
    # Create comparison figure
    fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 3, 6))
    
    # Handle case with batch_size=1
    if batch_size == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    
    # Plot standard samples
    for i in range(batch_size):
        img = samples_standard[i].permute(1, 2, 0).cpu().numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)  # Denormalize
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Standard #{i+1}")
        axes[0, i].axis('off')
    
    # Plot multi-anneal samples
    for i in range(batch_size):
        img = samples_multi[i].permute(1, 2, 0).cpu().numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)  # Denormalize
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"Multi-Anneal #{i+1}")
        axes[1, i].axis('off')
    
    # Add overall title with metrics
    plt.suptitle(f"Comparison: Standard vs Multi-Anneal\n"
                f"Standard: {standard_time:.2f}s, Diversity: {std_standard:.4f}\n"
                f"Multi-Anneal: {multi_time:.2f}s, Diversity: {std_multi:.4f}\n"
                f"Denoising Steps: {steps}, Anneals: {anneals_per_run}, Top-k: {top_k_percent}%",
                fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save figure
    comparison_path = output_path / f"comparison_steps{steps}_anneals{anneals_per_run}_topk{int(top_k_percent)}.png"
    plt.savefig(comparison_path)
    logger.info(f"Saved comparison to {comparison_path}")
    
    # Save normalized samples
    samples_standard_norm = (samples_standard + 1) / 2
    samples_multi_norm = (samples_multi + 1) / 2
    
    save_image(make_grid(samples_standard_norm, nrow=4),
              os.path.join(output_dir, "samples_standard.png"))
    save_image(make_grid(samples_multi_norm, nrow=4),
              os.path.join(output_dir, "samples_multi_anneal.png"))
    
    # Compile metrics
    metrics = {
        'standard_time': standard_time / batch_size,
        'multi_time': multi_time / batch_size,
        'standard_diversity': std_standard,
        'multi_diversity': std_multi,
        'time_difference_percent': ((multi_time - standard_time) / standard_time) * 100,
        'diversity_difference_percent': ((std_multi - std_standard) / std_standard) * 100
    }
    
    # Save metrics
    with open(os.path.join(output_dir, "comparison_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return samples_standard, samples_multi, metrics

def test_with_different_qubit_counts(checkpoint_path, qubit_counts, dataset_name, output_dir, 
                                   batch_size=16, steps=16, multi_anneal=False):
    """
    Test model with different qubit count configurations.
    
    Args:
        checkpoint_path: Path to model checkpoint
        qubit_counts: List of qubit counts to test with
        dataset_name: Name of dataset to test on
        output_dir: Output directory
        batch_size: Batch size for testing
        steps: Number of diffusion steps
        multi_anneal: Whether to use multi-anneal sampling
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test dataset
    test_loader, sample_shape = load_test_dataset(dataset_name, batch_size)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Test with each qubit count
    results = {}
    
    for qubit_count in qubit_counts:
        logger.info(f"Testing with {qubit_count} qubit configuration")
        
        # Load configuration for this qubit count
        config = load_config(None, qubit_count, sample_shape[1])
        
        # Ensure correct image size
        config['image_size'] = [sample_shape[1], sample_shape[2]]
        
        # Load model from checkpoint
        model = create_model_from_checkpoint(checkpoint_path, config, device)
        
        # Configure quantum sampler
        model = configure_quantum_sampler(model, config, multi_anneal)
        
        # Create output directory for this qubit count
        qubit_dir = output_path / f"qubit_{qubit_count}"
        qubit_dir.mkdir(exist_ok=True)
        
        # Save configuration
        with open(qubit_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Evaluate model
        metrics, test_images, reconstructions, samples = evaluate_model(
            model, test_loader, device, batch_size, steps)
        
        # Save samples
        samples_norm = (samples + 1) / 2
        save_image(make_grid(samples_norm, nrow=4),
                  os.path.join(qubit_dir, "samples.png"))
        
        # Save individual samples
        samples_dir = qubit_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        
        for i in range(batch_size):
            save_image(samples_norm[i],
                      os.path.join(samples_dir, f"sample_{i+1}.png"))
        
        # Save metrics
        with open(qubit_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Store results
        results[qubit_count] = metrics
    
    # Create comparison plot
    try:
        import matplotlib.pyplot as plt
        
        # Compare generation time
        plt.figure(figsize=(10, 6))
        qubit_counts_list = list(results.keys())
        gen_times = [results[qc]['generation_time'] for qc in qubit_counts_list]
        plt.bar([str(qc) for qc in qubit_counts_list], gen_times)
        plt.ylabel('Generation Time (s)')
        plt.xlabel('Qubit Count')
        plt.title('Generation Time vs Qubit Count')
        plt.savefig(os.path.join(output_dir, "qubit_generation_time.png"))
        
        # Compare sample diversity
        plt.figure(figsize=(10, 6))
        sample_div = [results[qc]['sample_diversity'] for qc in qubit_counts_list]
        plt.bar([str(qc) for qc in qubit_counts_list], sample_div)
        plt.ylabel('Sample Diversity')
        plt.xlabel('Qubit Count')
        plt.title('Sample Diversity vs Qubit Count')
        plt.savefig(os.path.join(output_dir, "qubit_sample_diversity.png"))
        
        # Compare PSNR
        plt.figure(figsize=(10, 6))
        psnr_values = [results[qc]['psnr'] for qc in qubit_counts_list]
        plt.bar([str(qc) for qc in qubit_counts_list], psnr_values)
        plt.ylabel('PSNR (dB)')
        plt.xlabel('Qubit Count')
        plt.title('Reconstruction PSNR vs Qubit Count')
        plt.savefig(os.path.join(output_dir, "qubit_psnr.png"))
    except ImportError:
        logger.warning("Matplotlib not available, skipping plot generation")
    
    # Save overall results
    with open(os.path.join(output_dir, "qubit_comparison_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Comparison across qubit counts completed")
    return results

def main():
    """Main testing function."""
    # Parse arguments
    args = parse_args()
    
    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        args.output_dir, 
        f"test_{args.dataset}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Save args for reference
    with open(os.path.join(output_dir, "test_args.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load dataset
    test_loader, sample_shape = load_test_dataset(args.dataset, args.batch_size)
    
    if args.compare_multi_anneal:
        # Compare standard vs multi-anneal sampling
        logger.info("Running comparison: standard vs multi-anneal sampling")
        
        # Load configuration
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
                if isinstance(config, list):
                    config = config[0]  # Take first config if list
        else:
            config = load_config(None, args.qubit_counts[0], sample_shape[1])
            config['image_size'] = [sample_shape[1], sample_shape[2]]
        
        # Create model from checkpoint
        model = create_model_from_checkpoint(args.checkpoint, config, device)
        
        # Run comparison
        compare_dir = os.path.join(output_dir, "multi_anneal_comparison")
        os.makedirs(compare_dir, exist_ok=True)
        
        _, _, comparison_metrics = compare_sampling_methods(
            model, config, device, compare_dir, 
            args.batch_size, args.steps,
            args.anneals_per_run, args.top_k_percent
        )
        
        logger.info("Comparison results:")
        for k, v in comparison_metrics.items():
            logger.info(f"  {k}: {v}")
    
    else:
        # Test with different qubit counts
        logger.info(f"Testing with qubit counts: {args.qubit_counts}")
        
        qubit_results = test_with_different_qubit_counts(
            args.checkpoint, 
            args.qubit_counts, 
            args.dataset, 
            output_dir, 
            args.batch_size, 
            args.steps, 
            multi_anneal=True  # Always use multi-anneal for comparison
        )
        
        # Summarize results
        logger.info("Testing completed. Summary of results:")
        for qc, metrics in qubit_results.items():
            logger.info(f"Qubit count: {qc}")
            for k, v in metrics.items():
                logger.info(f"  {k}: {v:.4f}")
    
    logger.info(f"All testing completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 