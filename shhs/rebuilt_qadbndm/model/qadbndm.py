"""
QADBNDM - Quantum-Assisted Deep Binary Neural Diffusion Model

This is the main model that integrates all components: encoder, decoder,
denoiser with DBNs, and quantum sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict, Union, Optional

from rebuilt_qadbndm.model.encoder.hierarchical_encoder import HierarchicalEncoder
from rebuilt_qadbndm.model.decoder.hierarchical_decoder import HierarchicalDecoder
from rebuilt_qadbndm.model.denoiser.dbn_denoiser import DBNDenoiser
from rebuilt_qadbndm.model.quantum.quantum_sampler import QuantumSampler
from rebuilt_qadbndm.model.quantum.latent_optimizer import LatentOptimizer

class QADBNDM(nn.Module):
    """
    Quantum-Assisted Deep Binary Neural Diffusion Model.
    
    Enhanced version with adaptive hierarchical structure, cross-level conditioning,
    and advanced content-aware bit allocation.
    """
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        hidden_channels: int = 128,
        latent_dims: List[int] = [32, 16, 8],
        scale_factors: List[int] = [8, 4, 2],
        max_levels: int = 4,
        min_levels: int = 2,
        use_attention: bool = True,
        use_adaptive_hierarchy: bool = True,
        use_adaptive_bit_allocation: bool = True,
        use_cross_level_attention: bool = True,
        use_cross_level_conditioning: bool = True,
        use_quantum: bool = False,
        binary_temp_schedule: bool = True,
        num_timesteps: int = 1000,
        dbn_hidden_units: Optional[List[int]] = None,
        schedule_types: Union[str, List[str]] = 'cosine',
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        """
        Initialize the QADBNDM model.
        
        Args:
            input_shape: Shape of input images (C,H,W)
            hidden_channels: Number of hidden channels
            latent_dims: List of latent dimensions for each level
            scale_factors: List of scaling factors for each level
            max_levels: Maximum number of hierarchical levels
            min_levels: Minimum number of hierarchical levels
            use_attention: Whether to use attention blocks
            use_adaptive_hierarchy: Whether to use adaptive hierarchy with learned level selection
            use_adaptive_bit_allocation: Whether to use content-aware bit allocation
            use_cross_level_attention: Whether to use cross-level attention for better information flow
            use_cross_level_conditioning: Whether to use cross-level conditioning in the diffusion process
            use_quantum: Whether to use quantum sampling
            binary_temp_schedule: Whether to anneal the binarization temperature
            num_timesteps: Number of timesteps in the diffusion process
            dbn_hidden_units: List of hidden units for each level's DBNs
            schedule_types: Type(s) of noise schedule ('linear', 'cosine', 'quadratic')
            device: Device to place model on
        """
        super().__init__()
        
        assert len(latent_dims) == len(scale_factors), "Must provide same number of latent dims and scale factors"
        assert len(latent_dims) <= max_levels, "Number of latent dims exceeds max_levels"
        
        self.input_shape = input_shape
        self.hidden_channels = hidden_channels
        self.latent_dims = latent_dims
        self.scale_factors = scale_factors
        self.max_levels = max_levels
        self.min_levels = min_levels
        self.use_attention = use_attention
        self.use_adaptive_hierarchy = use_adaptive_hierarchy
        self.use_adaptive_bit_allocation = use_adaptive_bit_allocation
        self.use_cross_level_attention = use_cross_level_attention
        self.use_cross_level_conditioning = use_cross_level_conditioning
        self.use_quantum = use_quantum
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Calculate spatial dimensions for each level
        spatial_dims = (input_shape[1], input_shape[2])
        
        # Create encoder
        self.encoder = HierarchicalEncoder(
            input_channels=input_shape[0],
            hidden_channels=hidden_channels,
            latent_dims=latent_dims,
            downscale_factors=scale_factors,
            max_levels=max_levels,
            min_levels=min_levels,
            use_attention=use_attention,
            spatial_dims=spatial_dims,
            initial_temperature=1.0 if binary_temp_schedule else 0.5,
            min_temperature=0.5,
            anneal_rate=0.9999 if binary_temp_schedule else 1.0,
            adaptive_hierarchy=use_adaptive_hierarchy,
            adaptive_bit_allocation=use_adaptive_bit_allocation
        )
        
        # Create decoder
        self.decoder = HierarchicalDecoder(
            output_channels=input_shape[0],
            hidden_channels=hidden_channels,
            latent_dims=latent_dims,
            upscale_factors=scale_factors,
            max_levels=max_levels,
            use_attention=use_attention,
            use_fusion=True,
            use_adaptive_fusion=True,
            use_cross_level_attention=use_cross_level_attention,
            final_refinement=True
        )
        
        # Calculate total binary latent size
        total_latent_size = self.encoder.get_total_latent_size()
        
        # Create DBN denoiser for hierarchical latent space with enhanced conditioning
        self.denoiser = DBNDenoiser(
            latent_sizes=self.encoder.calculate_latent_sizes(),
            hidden_units=dbn_hidden_units,
            num_timesteps=num_timesteps,
            cross_level_conditioning=use_cross_level_conditioning,
            schedule_types=schedule_types,
            device=device
        )
        
        # Optional quantum components
        if use_quantum:
            self.quantum_sampler = QuantumSampler(total_latent_size)
            self.latent_optimizer = LatentOptimizer(total_latent_size)
        else:
            self.quantum_sampler = None
            self.latent_optimizer = None
        
        # Loss weights for training
        self.register_buffer('reconstruction_weight', torch.tensor([1.0]))
        self.register_buffer('kl_weight', torch.tensor([0.1]))
        self.register_buffer('binary_weight', torch.tensor([0.05]))
        
        # Initialize binary distribution priors
        self.register_buffer('binary_prior', torch.tensor([0.5]))
        
        # Move to device
        self.to(device)
    
    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Encode input to hierarchical binary latent space.
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            List of binary latent tensors
        """
        return self.encoder(x)
    
    def decode(self, latents: List[torch.Tensor], level_indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        Decode binary latent representations to output space.
        
        Args:
            latents: List of binary latent tensors
            level_indices: Optional indices of active levels
            
        Returns:
            Decoded output tensor
        """
        return self.decoder(latents, level_indices)
    
    def denoise(
        self, 
        latents: List[torch.Tensor], 
        noise_level: Optional[float] = None,
        steps: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        Denoise binary latent representations.
        
        Args:
            latents: List of binary latent tensors
            noise_level: Optional noise level (0 to 1) to add before denoising
            steps: Optional number of denoising steps
            
        Returns:
            List of denoised binary latent tensors
        """
        return self.denoiser(
            latents,
            noise_level=noise_level,
            steps=steps,
            quantum_sampler=self.quantum_sampler if self.use_quantum else None
        )
    
    def sample(
        self, 
        batch_size: int = 1, 
        noise_level: float = 1.0, 
        steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Sample from the model by starting from noise and denoising.
        
        Args:
            batch_size: Number of samples to generate
            noise_level: Noise level (0 to 1) to start from
            steps: Optional number of denoising steps
            
        Returns:
            Generated samples
        """
        # Determine how many levels to use
        if self.use_adaptive_hierarchy:
            # When sampling, we use all base levels for better quality
            num_levels = self.min_levels
            active_levels = list(range(min(num_levels, len(self.latent_dims))))
        else:
            # Use all base levels in the static hierarchy
            num_levels = len(self.latent_dims)
            active_levels = list(range(num_levels))
        
        # Create initial random latents
        device = next(self.parameters()).device
        random_latents = []
        
        for i in active_levels:
            # Calculate shape for this level
            h = self.input_shape[1] // self.scale_factors[i]
            w = self.input_shape[2] // self.scale_factors[i]
            
            # Create random binary latent with values {0,1}
            random_latent = torch.randint(
                0, 2, (batch_size, self.latent_dims[i], h, w), 
                device=device
            ).float()
            
            random_latents.append(random_latent)
        
        # Denoise the random latents
        denoised_latents = self.denoise(
            random_latents,
            noise_level=noise_level,
            steps=steps
        )
        
        # Decode the denoised latents
        samples = self.decode(denoised_latents, active_levels)
        
        return samples
    
    def denoise_with_optimization(
        self, 
        latents: List[torch.Tensor], 
        noise_level: float, 
        steps: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        Denoise with latent space optimization for quantum resources.
        
        Args:
            latents: List of binary latent tensors
            noise_level: Noise level (0 to 1)
            steps: Optional number of denoising steps
            
        Returns:
            List of denoised binary latent tensors
        """
        if not self.use_quantum or self.latent_optimizer is None:
            return self.denoise(latents, noise_level, steps)
        
        # Use latent optimizer to determine optimal allocation of quantum resources
        optimization_result = self.latent_optimizer.optimize(
            latents=latents,
            noise_level=noise_level,
            dbns_per_run=self.denoiser.dbns_per_run,
            use_cross_level=self.use_cross_level_conditioning
        )
        
        # Configure quantum sampler with optimization results
        self.quantum_sampler.configure(
            num_anneals=optimization_result['num_anneals'],
            anneal_time=optimization_result['anneal_time'],
            level_priorities=optimization_result['level_priorities']
        )
        
        # Run denoising with optimized parameters
        return self.denoise(latents, noise_level, steps)
    
    def optimize_latent_space(self, latents: List[torch.Tensor]) -> dict:
        """
        Optimize latent space for better quantum resource allocation.
        
        Args:
            latents: List of binary latent tensors
            
        Returns:
            Optimization results
        """
        if not self.use_quantum or self.latent_optimizer is None:
            return {}
        
        return self.latent_optimizer.optimize(
            latents=latents,
            noise_level=0.5,  # Middle noise level for general optimization
            dbns_per_run=self.denoiser.dbns_per_run,
            use_cross_level=self.use_cross_level_conditioning
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        noise_level: Optional[float] = None,
        return_latents: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            noise_level: Optional noise level (0 to 1) to add during autoencoding
            return_latents: Whether to return latent representations
            
        Returns:
            Reconstructed tensor or tuple of (reconstructed tensor, latents)
        """
        # Encode to hierarchical binary latent space
        latents = self.encode(x)
        
        # Get active level indices based on returned latents
        active_level_indices = list(range(len(latents)))
        
        # Apply denoising if noise level is specified
        if noise_level is not None and noise_level > 0:
            latents = self.denoise(latents, noise_level)
        
        # Decode back to input space
        reconstruction = self.decode(latents, active_level_indices)
        
        if return_latents:
            return reconstruction, latents
        else:
            return reconstruction

    def update_temperature(self, epoch: int, max_epochs: int):
        """
        Update temperature for binarization based on training progress.
        Allows for curriculum learning on the binary representations.
        
        Args:
            epoch: Current epoch
            max_epochs: Maximum number of epochs
        """
        if not self.binary_temp_schedule:
            return
        
        # Cosine annealing schedule
        progress = epoch / max_epochs
        temp = 0.5 + 0.5 * (1 + math.cos(math.pi * progress))
        
        # Update encoder temperature
        with torch.no_grad():
            self.encoder.temperature.fill_(max(temp, 0.5))
            
    def set_binary_weight(self, weight: float):
        """
        Set weight for binary entropy regularization.
        
        Args:
            weight: New weight value
        """
        with torch.no_grad():
            self.binary_weight.fill_(weight)
    
    def set_kl_weight(self, weight: float):
        """
        Set weight for KL divergence term.
        
        Args:
            weight: New weight value
        """
        with torch.no_grad():
            self.kl_weight.fill_(weight) 