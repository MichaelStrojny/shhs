"""
Quantum module for QADBNDM

This module provides quantum-related components for the QADBNDM model,
including quantum sampling and latent optimization.
"""

from rebuilt_qadbndm.model.quantum.quantum_sampler import QuantumSampler
from rebuilt_qadbndm.model.quantum.latent_optimizer import LatentOptimizer

__all__ = ['QuantumSampler', 'LatentOptimizer']
