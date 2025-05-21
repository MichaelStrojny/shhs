"""
Latent Optimizer

This module implements latent space optimization for quantum resource allocation
in the QADBNDM model. It analyzes latent representations to determine optimal 
quantum annealing parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any
import math

class LatentOptimizer:
    """
    Optimizes quantum computing resources for latent space denoising.
    
    This class analyzes binary latent representations to determine the optimal
    allocation of quantum annealing resources for maximum denoising quality.
    """
    def __init__(
        self,
        latent_size: int,
        min_anneals: int = 50,
        max_anneals: int = 1000,
        min_anneal_time: float = 10.0,
        max_anneal_time: float = 100.0,
        complexity_threshold: float = 0.5,
        batch_factor: float = 2.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the latent optimizer.
        
        Args:
            latent_size: Maximum size of the latent space
            min_anneals: Minimum number of annealing runs per sample
            max_anneals: Maximum number of annealing runs per sample
            min_anneal_time: Minimum annealing time in microseconds
            max_anneal_time: Maximum annealing time in microseconds
            complexity_threshold: Threshold for determining complex vs. simple regions
            batch_factor: Factor for determining batch size based on complexity
            device: Device to use for tensor operations
        """
        self.latent_size = latent_size
        self.min_anneals = min_anneals
        self.max_anneals = max_anneals
        self.min_anneal_time = min_anneal_time
        self.max_anneal_time = max_anneal_time
        self.complexity_threshold = complexity_threshold
        self.batch_factor = batch_factor
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Precompute normalized scale for different latent sizes
        self.max_latent_scale = max(1.0, math.log10(latent_size) / 3)
    
    def _estimate_latent_complexity(self, latents: List[torch.Tensor]) -> Tuple[float, List[float]]:
        """
        Estimate the complexity of latent representations based on entropy and spatial structure.
        
        Args:
            latents: List of binary latent tensors
            
        Returns:
            Tuple of (overall complexity score, per-level complexity scores)
        """
        level_complexities = []
        
        for latent in latents:
            # Calculate binary entropy
            batch_size = latent.shape[0]
            
            # Flatten spatial dimensions
            latent_flat = latent.reshape(batch_size, latent.shape[1], -1)
            
            # Calculate average over batch
            latent_mean = latent_flat.mean(dim=0)
            
            # Calculate entropy: -p*log(p) - (1-p)*log(1-p)
            # Add small epsilon to prevent log(0)
            eps = 1e-8
            p = latent_mean.clamp(eps, 1 - eps)
            entropy = -p * torch.log2(p) - (1 - p) * torch.log2(1 - p)
            
            # Higher entropy means more uncertainty/complexity
            mean_entropy = entropy.mean().item()
            
            # Calculate spatial structure using gradient magnitude
            if latent.dim() > 3:  # Check if we have spatial dimensions
                # Calculate gradients in spatial dimensions
                dx = torch.abs(latent[:, :, :, 1:] - latent[:, :, :, :-1]).mean().item()
                dy = torch.abs(latent[:, :, 1:, :] - latent[:, :, :-1, :]).mean().item()
                spatial_complexity = (dx + dy) / 2
            else:
                spatial_complexity = 0.0
            
            # Calculate overall complexity for this level
            level_complexity = (mean_entropy + spatial_complexity) / 2
            level_complexities.append(level_complexity)
        
        # Overall complexity is weighted average of level complexities
        # With higher weight for coarser levels as they're more important
        weights = [1.0 / (i + 1) for i in range(len(level_complexities))]
        weight_sum = sum(weights)
        overall_complexity = sum(c * w for c, w in zip(level_complexities, weights)) / weight_sum
        
        return overall_complexity, level_complexities
    
    def _calculate_level_priorities(self, latents: List[torch.Tensor], level_complexities: List[float]) -> List[float]:
        """
        Calculate priority weights for each level based on complexity and importance.
        
        Args:
            latents: List of binary latent tensors
            level_complexities: Complexity score for each level
            
        Returns:
            List of priority weights for each level
        """
        # Calculate normalized sizes
        level_sizes = [latent.numel() / latent.shape[0] for latent in latents]
        max_size = max(level_sizes)
        norm_sizes = [size / max_size for size in level_sizes]
        
        # Coarser levels (earlier in list) have higher base priority
        base_priorities = [1.0 / (i + 1) for i in range(len(latents))]
        
        # Combine base priority with complexity and size
        priorities = []
        for i, (base_p, complexity, size) in enumerate(zip(base_priorities, level_complexities, norm_sizes)):
            # Priority formula: base_priority * (1 + complexity) * size
            # This gives higher priority to:
            # 1. Coarser levels (through base_priority)
            # 2. More complex levels (through complexity term)
            # 3. Larger levels (through size term)
            priority = base_p * (1 + complexity) * size
            priorities.append(priority)
        
        # Normalize priorities to sum to 1
        priority_sum = sum(priorities)
        if priority_sum > 0:
            priorities = [p / priority_sum for p in priorities]
        else:
            # If priorities sum to 0 (shouldn't happen), use uniform priorities
            priorities = [1.0 / len(latents) for _ in latents]
        
        return priorities
    
    def _calculate_optimal_anneals(
        self, 
        overall_complexity: float, 
        level_complexities: List[float],
        noise_level: float
    ) -> int:
        """
        Calculate the optimal number of annealing runs based on complexity and noise level.
        
        Args:
            overall_complexity: Overall complexity score
            level_complexities: Complexity score for each level
            noise_level: Current noise level (0 to 1)
            
        Returns:
            Number of annealing runs to perform
        """
        # Scale complexity by latent size
        scaled_complexity = overall_complexity * self.max_latent_scale
        
        # Higher noise levels and higher complexity require more anneals
        noise_factor = 0.5 + 0.5 * noise_level
        
        # Calculate optimal anneals
        optimal_anneals = self.min_anneals + (self.max_anneals - self.min_anneals) * scaled_complexity * noise_factor
        
        # Round to nearest 10
        optimal_anneals = int(round(optimal_anneals / 10) * 10)
        
        # Clamp to range
        optimal_anneals = max(self.min_anneals, min(self.max_anneals, optimal_anneals))
        
        return optimal_anneals
    
    def _calculate_optimal_anneal_time(
        self, 
        overall_complexity: float, 
        level_complexities: List[float],
        noise_level: float
    ) -> float:
        """
        Calculate the optimal annealing time based on complexity and noise level.
        
        Args:
            overall_complexity: Overall complexity score
            level_complexities: Complexity score for each level
            noise_level: Current noise level (0 to 1)
            
        Returns:
            Annealing time in microseconds
        """
        # Scale complexity by latent size
        scaled_complexity = overall_complexity * self.max_latent_scale
        
        # Higher noise levels and higher complexity require longer annealing times
        noise_factor = 0.5 + 0.5 * noise_level
        
        # Calculate optimal anneal time
        optimal_time = self.min_anneal_time + (self.max_anneal_time - self.min_anneal_time) * scaled_complexity * noise_factor
        
        # Round to nearest 5
        optimal_time = round(optimal_time / 5) * 5
        
        # Clamp to range
        optimal_time = max(self.min_anneal_time, min(self.max_anneal_time, optimal_time))
        
        return optimal_time
    
    def _optimal_dbns_per_run(
        self, 
        overall_complexity: float, 
        level_complexities: List[float],
        dbns_per_run: int,
        use_cross_level: bool
    ) -> int:
        """
        Calculate the optimal number of DBNs to process in a single annealer run.
        
        Args:
            overall_complexity: Overall complexity score
            level_complexities: Complexity score for each level
            dbns_per_run: Maximum number of DBNs per run
            use_cross_level: Whether cross-level conditioning is used
            
        Returns:
            Optimal number of DBNs per run
        """
        # Base number is the maximum provided
        base_dbns = dbns_per_run
        
        # Reduce based on complexity - more complex latents need more focused annealing
        complexity_factor = 1.0 - overall_complexity * 0.5  # Range: 0.5 to 1.0
        
        # Cross-level conditioning adds overhead, reduce batch size
        cross_level_factor = 0.7 if use_cross_level else 1.0
        
        # Calculate optimal number
        optimal_dbns = max(1, int(base_dbns * complexity_factor * cross_level_factor))
        
        return optimal_dbns
    
    def optimize(
        self, 
        latents: List[torch.Tensor],
        noise_level: float = 0.5,
        dbns_per_run: int = 10,
        use_cross_level: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize quantum annealing parameters for the given latent representation.
        
        Args:
            latents: List of binary latent tensors
            noise_level: Current noise level (0 to 1)
            dbns_per_run: Maximum number of DBNs to process in a single annealer run
            use_cross_level: Whether cross-level conditioning is used
            
        Returns:
            Dictionary of optimized parameters
        """
        # Estimate latent complexity
        overall_complexity, level_complexities = self._estimate_latent_complexity(latents)
        
        # Calculate level priorities
        level_priorities = self._calculate_level_priorities(latents, level_complexities)
        
        # Calculate optimal number of anneals
        num_anneals = self._calculate_optimal_anneals(overall_complexity, level_complexities, noise_level)
        
        # Calculate optimal annealing time
        anneal_time = self._calculate_optimal_anneal_time(overall_complexity, level_complexities, noise_level)
        
        # Calculate optimal DBNs per run
        optimal_dbns = self._optimal_dbns_per_run(overall_complexity, level_complexities, dbns_per_run, use_cross_level)
        
        # Return optimization results
        return {
            "num_anneals": num_anneals,
            "anneal_time": anneal_time,
            "level_priorities": level_priorities,
            "overall_complexity": overall_complexity,
            "level_complexities": level_complexities,
            "optimal_dbns_per_run": optimal_dbns
        } 