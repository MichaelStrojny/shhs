"""
Quantum-Assisted Deep Binary Neural Diffusion Model (QADBNDM)

This package implements a diffusion model that operates in binary latent space
with adaptive hierarchical structure and cross-level conditioning, which can
leverage quantum annealing for improved sampling.
"""

__version__ = "0.2.0"

from rebuilt_qadbndm.model.qadbndm import QADBNDM

__all__ = ['QADBNDM'] 