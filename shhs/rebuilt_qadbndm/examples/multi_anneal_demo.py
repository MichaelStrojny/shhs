#!/usr/bin/env python
"""
Multi-Anneal Demonstration

This script demonstrates the usage of the multi-anneal feature in the QADBNDM model
to generate high-quality images through multiple annealing runs with selection.
"""

import os
import argparse
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from time import time
from pathlib import Path

from rebuilt_qadbndm.model.qadbndm import QADBNDM
from rebuilt_qadbndm.model.quantum.quantum_sampler import QuantumSampler

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Demonstrate multi-anneal sampling with QADBNDM")
    parser.add_argument("--config", type=str, help="Path to configuration file (optional)")
    parser.add_argument("--qubit-count", type=int, default=2048, 
                      help="Qubit count for model configuration (ignored if --config is used)")
    parser.add_argument("--batch-size", type=int, default=4, 
                      help="Batch size for generation")
    parser.add_argument("--image-size", type=int, default=64,
                      help="Size of generated images (ignored if --config is used)")
    parser.add_argument("--output-dir", type=str, default="outputs",
                      help="Directory to save outputs")
    parser.add_argument("--compare", action="store_true",
                      help="Compare multi-anneal vs standard mode")
    parser.add_argument("--steps", type=int, default=16,
                      help="Number of denoising steps")
    parser.add_argument("--anneals-per-run", type=int, default=20,
                      help="Number of anneals per run (for multi-anneal mode)")
    parser.add_argument("--top-k-percent", type=float, default=20.0,
                      help="Percentage of top solutions to select (for multi-anneal mode)")
    return parser.parse_args()

def load_config(config_path=None, qubit_count=2048, image_size=64):
    """
    Load model configuration from file or use default for specified qubit count.
    
    Args:
        config_path: Path to configuration file
        qubit_count: Qubit count for default configuration
        image_size: Image size for default configuration
        
    Returns:
        Model configuration dictionary
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            configs = json.load(f)
            
        # Find a config matching our image size
        for config in configs:
            if config['image_size'][0] == image_size:
                return config
        
        # If no matching config found, use first one
        return configs[0]
    
    # Default configurations for different qubit counts
    # These are simplified versions - the generate_quantum_configs script produces more optimized ones
    default_configs = {
        2048: {
            "qubit_count": 2048,
            "image_size": [image_size, image_size],
            "model_parameters": {
                "latent_dims": [16, 12, 8],
                "scale_factors": [8, 4, 2],
                "hidden_channels": 128,
                "use_attention": True,
                "dbns_per_run": 4
            },
            "quantum_parameters": {
                "multi_anneal": True,
                "anneals_per_run": 20,
                "top_k_percent": 20.0,
                "target_batch_size": 4
            }
        },
        5760: {
            "qubit_count": 5760,
            "image_size": [image_size, image_size],
            "model_parameters": {
                "latent_dims": [24, 16, 12],
                "scale_factors": [8, 4, 2],
                "hidden_channels": 128,
                "use_attention": True,
                "dbns_per_run": 8
            },
            "quantum_parameters": {
                "multi_anneal": True,
                "anneals_per_run": 30,
                "top_k_percent": 15.0,
                "target_batch_size": 4
            }
        },
        8192: {
            "qubit_count": 8192,
            "image_size": [image_size, image_size],
            "model_parameters": {
                "latent_dims": [32, 24, 16],
                "scale_factors": [8, 4, 2],
                "hidden_channels": 128,
                "use_attention": True,
                "dbns_per_run": 10
            },
            "quantum_parameters": {
                "multi_anneal": True,
                "anneals_per_run": 40,
                "top_k_percent": 10.0,
                "target_batch_size": 4
            }
        }
    }
    
    # Use configuration for specified qubit count, or fallback to smallest
    return default_configs.get(qubit_count, default_configs[2048])

def create_model_from_config(config, device, custom_anneals=None, custom_top_k=None):
    """
    Create a QADBNDM model from configuration.
    
    Args:
        config: Model configuration dictionary
        device: Device to place model on
        custom_anneals: Override anneals_per_run if provided
        custom_top_k: Override top_k_percent if provided
        
    Returns:
        Instantiated QADBNDM model
    """
    # Create model
    model = QADBNDM(
        input_shape=(3, config['image_size'][0], config['image_size'][1]),
        hidden_channels=config['model_parameters']['hidden_channels'],
        latent_dims=config['model_parameters']['latent_dims'],
        scale_factors=config['model_parameters']['scale_factors'],
        use_attention=config['model_parameters']['use_attention'],
        use_quantum=True,
        binary_temp_schedule=True,
        device=device
    )
    
    # Configure quantum sampler
    anneals = custom_anneals or config['quantum_parameters']['anneals_per_run']
    top_k = custom_top_k or config['quantum_parameters']['top_k_percent']
    
    model.quantum_sampler = QuantumSampler(
        latent_size=model.encoder.get_total_latent_size(),
        max_batch_size=config['model_parameters']['dbns_per_run'],
        multi_anneal=config['quantum_parameters']['multi_anneal'],
        anneals_per_run=anneals,
        top_k_percent=top_k,
        target_batch_size=config['quantum_parameters']['target_batch_size'],
        device=device
    )
    
    return model

def compare_sampling_methods(config, batch_size, steps, output_dir, anneals_per_run, top_k_percent):
    """
    Compare standard sampling with multi-anneal sampling.
    
    Args:
        config: Model configuration
        batch_size: Number of images to generate
        steps: Number of denoising steps
        output_dir: Directory to save outputs
        anneals_per_run: Number of anneals per run
        top_k_percent: Percentage of top solutions to select
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating models for comparison...")
    print(f"Device: {device}")
    
    # Create model with multi-anneal sampling
    model_multi = create_model_from_config(config, device, anneals_per_run, top_k_percent)
    
    # Create model with standard sampling (same configuration but multi_anneal=False)
    config_standard = config.copy()
    config_standard['quantum_parameters'] = config_standard['quantum_parameters'].copy()
    config_standard['quantum_parameters']['multi_anneal'] = False
    model_standard = create_model_from_config(config_standard, device)
    
    # Generate samples with both models
    print(f"Generating samples with standard quantum sampling...")
    start_time = time()
    samples_standard = model_standard.generate(batch_size=batch_size, steps=steps)
    standard_time = time() - start_time
    print(f"Standard sampling time: {standard_time:.2f}s")
    
    print(f"Generating samples with multi-anneal sampling...")
    start_time = time()
    samples_multi = model_multi.generate(batch_size=batch_size, steps=steps)
    multi_time = time() - start_time
    print(f"Multi-anneal sampling time: {multi_time:.2f}s")
    
    # Calculate perceptual quality (using variance as a simple proxy)
    # In a real application, you would use a proper perceptual metric
    std_standard = samples_standard.std(dim=[0, 2, 3]).mean().item()
    std_multi = samples_multi.std(dim=[0, 2, 3]).mean().item()
    
    # Create comparison figure
    fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 3, 6))
    
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
    print(f"Saved comparison to {comparison_path}")
    
    # Create HTML report with results
    report_path = output_path / "comparison_report.html"
    with open(report_path, 'w') as f:
        f.write(f"""
        <html>
        <head>
            <title>QADBNDM Multi-Anneal Comparison</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .comparison {{ text-align: center; margin: 30px 0; }}
                .result {{ font-weight: bold; color: #0066cc; }}
                .improvement {{ color: green; }}
                .regression {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>QADBNDM Multi-Anneal Sampling Comparison</h1>
            <p>This report compares standard quantum sampling with multi-anneal sampling for image generation.</p>
            
            <h2>Configuration</h2>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Image Size</td><td>{config['image_size'][0]}x{config['image_size'][1]}</td></tr>
                <tr><td>Qubit Count</td><td>{config['qubit_count']}</td></tr>
                <tr><td>Denoising Steps</td><td>{steps}</td></tr>
                <tr><td>Anneals Per Run</td><td>{anneals_per_run}</td></tr>
                <tr><td>Top-k Percent</td><td>{top_k_percent}%</td></tr>
                <tr><td>DBNs Per Run</td><td>{config['model_parameters']['dbns_per_run']}</td></tr>
                <tr><td>Latent Dimensions</td><td>{config['model_parameters']['latent_dims']}</td></tr>
            </table>
            
            <h2>Performance Results</h2>
            <table>
                <tr><th>Metric</th><th>Standard Sampling</th><th>Multi-Anneal Sampling</th><th>Difference</th></tr>
                <tr>
                    <td>Generation Time</td>
                    <td>{standard_time:.2f}s</td>
                    <td>{multi_time:.2f}s</td>
                    <td class="{'improvement' if multi_time < standard_time else 'regression'}">
                        {((multi_time - standard_time) / standard_time * 100):.1f}%
                    </td>
                </tr>
                <tr>
                    <td>Sample Diversity</td>
                    <td>{std_standard:.4f}</td>
                    <td>{std_multi:.4f}</td>
                    <td class="{'improvement' if std_multi > std_standard else 'regression'}">
                        {((std_multi - std_standard) / std_standard * 100):.1f}%
                    </td>
                </tr>
            </table>
            
            <div class="comparison">
                <h2>Visual Comparison</h2>
                <img src="{comparison_path.name}" alt="Comparison of standard vs multi-anneal sampling" style="max-width: 100%;" />
            </div>
            
            <h2>Conclusion</h2>
            <p class="result">
                Multi-anneal sampling {'improved' if std_multi > std_standard else 'did not improve'} sample diversity 
                by {abs((std_multi - std_standard) / std_standard * 100):.1f}% 
                {'with' if multi_time > standard_time else 'without'} a time penalty of 
                {abs((multi_time - standard_time) / standard_time * 100):.1f}%.
            </p>
        </body>
        </html>
        """)
    
    print(f"Saved HTML report to {report_path}")

def generate_with_multi_anneal(config, batch_size, steps, output_dir, anneals_per_run, top_k_percent):
    """
    Generate images using multi-anneal sampling.
    
    Args:
        config: Model configuration
        batch_size: Number of images to generate
        steps: Number of denoising steps
        output_dir: Directory to save outputs
        anneals_per_run: Number of anneals per run
        top_k_percent: Percentage of top solutions to select
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating model...")
    print(f"Device: {device}")
    print(f"Qubit count: {config['qubit_count']}")
    print(f"Image size: {config['image_size'][0]}x{config['image_size'][1]}")
    print(f"Latent dimensions: {config['model_parameters']['latent_dims']}")
    print(f"Scale factors: {config['model_parameters']['scale_factors']}")
    print(f"DBNs per run: {config['model_parameters']['dbns_per_run']}")
    
    # Create model
    model = create_model_from_config(config, device, anneals_per_run, top_k_percent)
    
    # Generate samples
    print(f"\nGenerating {batch_size} samples with {steps} denoising steps...")
    print(f"Multi-anneal parameters: {anneals_per_run} anneals per run, {top_k_percent}% top-k")
    
    start_time = time()
    samples = model.generate(batch_size=batch_size, steps=steps)
    generation_time = time() - start_time
    
    print(f"Generation completed in {generation_time:.2f}s ({generation_time/batch_size:.2f}s per sample)")
    
    # Create figure
    fig, axes = plt.subplots(1, batch_size, figsize=(batch_size * 3, 3))
    if batch_size == 1:
        axes = [axes]
    
    # Plot samples
    for i in range(batch_size):
        img = samples[i].permute(1, 2, 0).cpu().numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)  # Denormalize
        axes[i].imshow(img)
        axes[i].set_title(f"Sample {i+1}")
        axes[i].axis('off')
    
    # Add overall title
    plt.suptitle(f"QADBNDM Multi-Anneal Samples\n"
                f"Steps: {steps}, Anneals: {anneals_per_run}, Top-k: {top_k_percent}%\n"
                f"Generation Time: {generation_time:.2f}s ({generation_time/batch_size:.2f}s per sample)",
                fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save figure
    samples_path = output_path / f"samples_steps{steps}_anneals{anneals_per_run}_topk{int(top_k_percent)}.png"
    plt.savefig(samples_path)
    print(f"Saved samples to {samples_path}")
    
    # Save individual samples
    for i in range(batch_size):
        img = samples[i].permute(1, 2, 0).cpu().numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)  # Denormalize
        img = (img * 255).astype(np.uint8)
        sample_path = output_path / f"sample_{i+1}_steps{steps}_anneals{anneals_per_run}_topk{int(top_k_percent)}.png"
        plt.imsave(sample_path, img)

def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config, args.qubit_count, args.image_size)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration for reference
    config_path = os.path.join(args.output_dir, "used_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    if args.compare:
        # Compare standard vs multi-anneal
        compare_sampling_methods(
            config, 
            args.batch_size, 
            args.steps, 
            args.output_dir,
            args.anneals_per_run,
            args.top_k_percent
        )
    else:
        # Just generate with multi-anneal
        generate_with_multi_anneal(
            config,
            args.batch_size,
            args.steps,
            args.output_dir,
            args.anneals_per_run,
            args.top_k_percent
        )

if __name__ == "__main__":
    main() 