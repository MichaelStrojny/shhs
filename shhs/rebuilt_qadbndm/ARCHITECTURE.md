# QADBNDM Architecture Overview

## Core Components

The rebuilt QADBNDM (Quantum-Assisted Deep Binary Neural Diffusion Model) consists of the following key components:

### 1. Hierarchical Encoder

Located in: `model/encoder/hierarchical_encoder.py`

The encoder transforms input data (like images) into a multi-level binary latent representation. Each level captures patterns at different scales:
- Higher levels (coarser): Capture broad, structural features
- Lower levels (finer): Capture detailed features

The encoder uses:
- Residual blocks for feature extraction
- Attention mechanisms to capture long-range dependencies
- Straight-through estimator for differentiable binarization

### 2. Hierarchical Decoder

Located in: `model/decoder/hierarchical_decoder.py`

The decoder reconstructs the original data from the binary latent representation. It:
- Processes each level from coarsest to finest
- Includes optional fusion between levels for better reconstruction
- Uses upsampling blocks and attention mechanisms for quality

### 3. DBN Denoiser

Located in: `model/denoiser/dbn_denoiser.py`

The denoiser uses Deep Belief Networks (DBNs) to denoise the binary latent representation:
- One DBN per diffusion timestep
- Separate DBNs for each level of the hierarchy
- Implements diffusion-based progressive denoising
- Can use either standard Gibbs sampling or quantum sampling

### 4. Quantum Sampler

Located in: `model/quantum/quantum_sampler.py`

The quantum sampler uses quantum annealing to sample from the DBNs:
- Converts DBNs to Binary Quadratic Models (BQMs)
- Uses D-Wave's quantum annealer or Neal's simulated annealer
- Can batch multiple DBNs in a single annealing run for efficiency
- Specialized for binary latent spaces

### 5. Latent Space Optimizer

Located in: `model/quantum/latent_optimizer.py`

The latent optimizer balances reconstruction quality with computational efficiency:
- Optimizes the trade-off between latent space size and number of DBNs per annealer run
- Estimates reconstruction quality, denoising quality, and quantum efficiency
- Allocates latent budget across hierarchical levels
- Computes appropriate downscale factors

### 6. Main QADBNDM Model

Located in: `model/qadbndm.py`

The main model class integrates all components:
- Manages the flow between all components
- Provides high-level API for encoding, denoising, and decoding
- Supports latent space optimization
- Implements forward pass for end-to-end processing

## Data Flow

1. Input data → Hierarchical Encoder → Binary latent representation (multiple levels)
2. Binary latent → (Optional) Add noise → DBN Denoiser → Denoised binary latent
3. Denoised binary latent → Hierarchical Decoder → Reconstructed output

During denoising, the quantum sampler assists the DBN denoiser with more efficient sampling.

## Key Architectural Decisions

1. **Hierarchical Representation**: Allows capturing patterns at multiple scales and better reconstruction quality

2. **Binary Latent Space**: Makes the model more amenable to quantum computing approaches and reduces memory requirements

3. **One DBN per Timestep**: Enables specialized modeling of the denoising process at each noise level

4. **Quantum Sampling**: Provides more efficient exploration of the solution space for DBNs

5. **Latent Optimization**: Balances model quality with computational efficiency

## Extensibility

The modular architecture makes it easy to extend or modify the model:

- Swap out the encoder/decoder with different architectures
- Replace the quantum sampler with alternative sampling methods
- Implement different denoising processes
- Add new optimization strategies

## Dependencies

- PyTorch: Core deep learning framework
- NumPy: Numerical computing
- D-Wave Neal: Simulated quantum annealing
- D-Wave Ocean SDK (optional): For real quantum hardware
- Matplotlib, tqdm: Visualization and progress tracking 