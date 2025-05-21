#!/usr/bin/env python3
"""
Quantum Configuration Generator

Generates optimized configurations for quantum-assisted model runs
based on available qubit count.
"""

import argparse
import math
import json
import os
from typing import List, Dict, Any, Tuple
import sys

def calculate_model_parameters(qubit_count: int) -> Dict[str, Any]:
    """
    Calculate optimal model parameters for a given qubit count.
    
    Args:
        qubit_count: Number of available qubits
        
    Returns:
        Dictionary of optimized parameters
    """
    # Safety margin to account for overhead in annealer embedding
    effective_qubits = int(qubit_count * 0.85)
    
    # Start with default configuration
    config = {
        "latent_dims": [32, 16, 8],
        "scale_factors": [8, 4, 2],
        "hidden_channels": 128,
        "max_levels": 4,
        "min_levels": 2,
        "dbns_per_run": 5,
        "num_anneals": 100,
        "annealing_time": 20.0,
        "use_quantum": True,
        "use_adaptive_hierarchy": True,
        "use_adaptive_bit_allocation": True,
        "use_cross_level_conditioning": True,
        "use_cross_level_attention": True,
        "schedule_types": ["cosine", "linear", "quadratic"],
        "multi_anneal": True
    }
    
    # Scale configuration based on qubit count
    if qubit_count <= 2048:
        # Small configuration for smaller QPUs
        config["latent_dims"] = [16, 8, 4]
        config["hidden_channels"] = 64
        config["dbns_per_run"] = 3
    elif qubit_count <= 5000:
        # Medium configuration
        config["latent_dims"] = [24, 12, 6]
        config["hidden_channels"] = 96
        config["dbns_per_run"] = 4
    elif qubit_count <= 7000:
        # Large configuration
        config["latent_dims"] = [32, 16, 8]
        config["hidden_channels"] = 128
        config["dbns_per_run"] = 5
    else:
        # Very large configuration
        config["latent_dims"] = [48, 24, 12, 6]
        config["scale_factors"] = [16, 8, 4, 2]
        config["hidden_channels"] = 192
        config["max_levels"] = 5
        config["min_levels"] = 3
        config["dbns_per_run"] = 8
    
    # Calculate DBN hidden dimensions based on visible dimensions
    # Add some safety factor to ensure we stay within qubit limits
    total_visible_bits = sum(config["latent_dims"])
    max_hidden_factor = effective_qubits / (2 * total_visible_bits * config["dbns_per_run"])
    
    # Limit to reasonable range
    hidden_factor = min(2.0, max(0.5, max_hidden_factor))
    config["hidden_factor"] = hidden_factor
    
    # Calculate dbn_hidden_units
    config["dbn_hidden_units"] = [int(dim * hidden_factor) for dim in config["latent_dims"]]
    
    # Adjust annealing parameters for QPU size
    if qubit_count > 5000:
        config["num_anneals"] = 200
        config["annealing_time"] = 50.0
    
    # Calculate total qubit usage estimate
    total_visible = sum(config["latent_dims"])
    total_hidden = sum(config["dbn_hidden_units"])
    estimated_qubits = (total_visible + total_hidden) * config["dbns_per_run"]
    config["estimated_qubits"] = estimated_qubits
    config["qubit_utilization"] = estimated_qubits / qubit_count
    
    return config

def generate_example_code(config: Dict[str, Any]) -> str:
    """
    Generate example code for model instantiation with the given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        String with example code
    """
    code = """
import torch
from rebuilt_qadbndm.model.qadbndm import QADBNDM

# Create model with optimized quantum configuration
model = QADBNDM(
    input_shape=(3, 64, 64),
    hidden_channels={hidden_channels},
    latent_dims={latent_dims},
    scale_factors={scale_factors},
    max_levels={max_levels},
    min_levels={min_levels},
    use_attention=True,
    use_adaptive_hierarchy={use_adaptive_hierarchy},
    use_adaptive_bit_allocation={use_adaptive_bit_allocation},
    use_cross_level_attention={use_cross_level_attention},
    use_cross_level_conditioning={use_cross_level_conditioning},
    use_quantum={use_quantum},
    dbn_hidden_units={dbn_hidden_units},
    schedule_types={schedule_types},
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# Configure quantum sampling
if model.quantum_sampler is not None:
    model.quantum_sampler.configure(
        num_anneals={num_anneals},
        anneal_time={annealing_time},
        max_batch_size={dbns_per_run},
        multi_anneal={multi_anneal}
    )

# Generate samples
samples = model.sample(batch_size=4, noise_level=1.0)
""".format(**config)
    
    return code

def main():
    parser = argparse.ArgumentParser(description="Generate optimized quantum configurations")
    parser.add_argument("--qubit-counts", nargs="+", type=int, default=[2048, 5760, 8192],
                        help="List of qubit counts to generate configurations for")
    parser.add_argument("--output-dir", type=str, default="configs",
                        help="Directory to save configuration files")
    parser.add_argument("--generate-examples", action="store_true",
                        help="Generate example code for each configuration")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Generate configurations for each qubit count
    for qubit_count in args.qubit_counts:
        print(f"Generating configuration for {qubit_count} qubits...")
        config = calculate_model_parameters(qubit_count)
        
        # Save configuration to JSON file
        output_file = os.path.join(args.output_dir, f"quantum_config_{qubit_count}.json")
        with open(output_file, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"  - Estimated qubit usage: {config['estimated_qubits']:.0f} ({config['qubit_utilization']*100:.1f}%)")
        print(f"  - Configuration saved to {output_file}")
        
        # Generate example code if requested
        if args.generate_examples:
            example_file = os.path.join(args.output_dir, f"example_{qubit_count}.py")
            with open(example_file, "w") as f:
                f.write(generate_example_code(config))
            print(f"  - Example code saved to {example_file}")
    
    print("\nConfiguration generation complete!")

if __name__ == "__main__":
    main() 