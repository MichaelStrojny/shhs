# QADBNDM: Quantum-Assisted Deep Binary Neural Diffusion Model

This repository contains the implementation of the enhanced QADBNDM, a diffusion model that operates in binary latent space with adaptive hierarchical structure and cross-level conditioning that can leverage quantum annealing for improved sampling.

## Overview

QADBNDM combines several key technologies:

1. **Binary Latent Space**: Represents images in a hierarchical binary latent space, making it suitable for both classical and quantum processing.
2. **Deep Belief Networks (DBNs)**: Used for modeling binary distributions and denoising binary latent representations.
3. **Quantum Annealing**: Leverages D-Wave Neal's simulated quantum annealing to improve sampling efficiency.
4. **Hierarchical Structure**: Represents data at multiple scales for better reconstruction quality.
5. **Cross-Level Conditioning**: Enhances information flow between hierarchical levels.
6. **Adaptive Bit Allocation**: Dynamically allocates bits based on content complexity.

## Key Features

- **Adaptive Hierarchical Levels**: Learns optimal hierarchy structure based on content complexity
- **Cross-Level Conditioning**: Improves diffusion process with information flow between levels
- **Content-Aware Bit Allocation**: Allocates more bits to complex regions requiring detailed representation
- **Cross-Level Attention**: Enhances information flow between levels during encoding and decoding
- **Multiple Noise Schedules**: Supports level-specific noise scheduling for optimal diffusion
- **Quantum-Accelerated Sampling**: Optional quantum annealing for improved exploration of binary space
- **Memory Efficiency**: Binary representations are more compact than continuous alternatives
- **Fully Differentiable**: End-to-end training despite discrete latent space

## Enhanced Features

### Adaptive Hierarchical Structure

The model dynamically learns the optimal hierarchy structure:

- **Learned Level Selection**: Determines optimal number and configuration of levels
- **Dynamic Level Gating**: Uses Gumbel-softmax for differentiable level selection
- **Input-Dependent Hierarchy**: Adapts structure based on input complexity
- **Cross-Level Fusion**: Better information flow between levels

### Content-Aware Bit Allocation

Optimizes bit allocation based on content complexity:

- **Complexity Estimators**: Learns to identify regions requiring more detailed representation
- **Dynamic Bit Weighting**: Assigns bit importance based on learned and content-based factors
- **Spatial Structure Processing**: Enhances spatial correlations in binary space
- **Local Variation Analysis**: Uses gradient information to identify complex regions

### Cross-Level Diffusion Process

Enhances the diffusion process with level dependencies:

- **Level-Specific Schedules**: Different noise schedules per hierarchical level
- **Cross-Level Conditioning**: Conditions fine level diffusion on coarser levels
- **Conditional DBNs**: Deep Belief Networks with explicit cross-level connections
- **Enhanced Noise Modeling**: Models correlations between levels during noising

### Multi-Anneal Sampling

The model supports multiple anneals per run with selection of top results:

- **Multiple Anneals**: Runs N anneals per annealer run and selects top k% results
- **Batch Size Optimization**: Automatically calculates number of anneals needed for generating M images
- **Energy-Based Selection**: Selects samples based on energy (lower is better)
- **Configurable Parameters**: Can adjust top-k percentage, number of anneals, etc.

### Quantum Configuration Generator

A script generates optimized configurations for different qubit counts:

- **Hierarchical Structure Optimization**: Determines optimal latent space size and structure
- **DBNs per Run Optimization**: Calculates how many DBNs to run per annealer invocation
- **Resource-Aware**: Optimizes for different hardware constraints (2048, 5436, 5760, 8192 qubits)
- **Example Code Generation**: Creates ready-to-use model instantiation code

## Usage

### Basic Usage

```python
import torch
from rebuilt_qadbndm.model.qadbndm import QADBNDM

# Create model with enhanced features
model = QADBNDM(
    input_shape=(3, 32, 32),
    hidden_channels=128,
    latent_dims=[32, 16, 8],
    scale_factors=[8, 4, 2],
    max_levels=4,
    min_levels=2,
    use_attention=True,
    use_adaptive_hierarchy=True,
    use_adaptive_bit_allocation=True,
    use_cross_level_attention=True,
    use_cross_level_conditioning=True,
    use_quantum=True
)

# Encode an image to binary latent space
latents = model.encode(image)

# Generate new images
samples = model.sample(batch_size=4, noise_level=1.0)
```

### Using Adaptive Hierarchy and Cross-Level Conditioning

```python
from rebuilt_qadbndm.model.qadbndm import QADBNDM

# Create model with adaptive hierarchy
model = QADBNDM(
    input_shape=(3, 64, 64),
    hidden_channels=128,
    latent_dims=[32, 16, 8],
    scale_factors=[8, 4, 2],
    max_levels=5,                   # Maximum number of levels
    min_levels=2,                   # Minimum number of levels
    use_adaptive_hierarchy=True,    # Enable adaptive hierarchy
    use_adaptive_bit_allocation=True,  # Enable content-aware bit allocation
    use_cross_level_attention=True,    # Enable cross-level attention
    use_cross_level_conditioning=True  # Enable cross-level conditioning in diffusion
)

# Encode input with learned level selection
latents = model.encode(image)

# Forward pass with noise
reconstruction, latents = model.forward(image, noise_level=0.2, return_latents=True)
```

### Using Multi-Anneal Quantum Sampling with Cross-Level Conditioning

```python
from rebuilt_qadbndm.model.qadbndm import QADBNDM
from rebuilt_qadbndm.model.quantum.quantum_sampler import QuantumSampler

# Create model with cross-level conditioning
model = QADBNDM(
    input_shape=(3, 64, 64),
    hidden_channels=128,
    latent_dims=[32, 16, 8],
    scale_factors=[8, 4, 2],
    use_cross_level_conditioning=True,
    schedule_types=['cosine', 'linear', 'quadratic'],  # Level-specific schedules
    use_quantum=True
)

# Configure quantum sampler with multi-anneal
model.quantum_sampler = QuantumSampler(
    latent_size=model.encoder.get_total_latent_size(),
    max_batch_size=5,  # DBNs per annealer run
    multi_anneal=True,
    anneals_per_run=10,
    top_k_percent=20,
    target_batch_size=4
)

# Generate samples with optimized quantum resources
samples = model.sample(batch_size=4, noise_level=1.0)

# Use latent space optimization for quantum resources
optimized_config = model.optimize_latent_space(latents)
```

### Generating Optimized Configurations

```bash
# Generate optimized configurations for default qubit counts
python -m rebuilt_qadbndm.scripts.generate_quantum_configs

# Specify custom qubit counts
python -m rebuilt_qadbndm.scripts.generate_quantum_configs --qubit-counts 2048 5760 8192

# Generate example code
python -m rebuilt_qadbndm.scripts.generate_quantum_configs --generate-examples
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/rebuilt_qadbndm.git
cd rebuilt_qadbndm

# Install dependencies
pip install -e .
```

## Dependencies

- PyTorch (>= 1.9.0)
- NumPy (>= 1.19.0)
- D-Wave Neal (for simulated quantum annealing)
- D-Wave Ocean SDK (optional, for real quantum hardware)

## Citation

If you use this code in your research, please cite our paper:

```
@article{qadbndm2023,
  title={Quantum-Assisted Deep Binary Neural Diffusion Models with Adaptive Hierarchy and Cross-Level Conditioning},
  author={Your Name},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 