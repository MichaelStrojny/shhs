"""
DBN Denoiser

This module implements the denoising process for binary latent representations
using Deep Belief Networks (DBNs), with one DBN per timestep.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any
import time

class DeepBeliefNetwork(nn.Module):
    """
    Deep Belief Network for modeling binary distributions.
    
    This is used for learning the denoising process in the diffusion model.
    """
    def __init__(
        self,
        visible_dim: int,
        hidden_dim: int,
        timestep: int = 0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the DBN.
        
        Args:
            visible_dim: Dimension of visible (input) layer
            hidden_dim: Dimension of hidden layer
            timestep: Which timestep in the diffusion process this DBN models
            device: Device to place model on
        """
        super().__init__()
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.timestep = timestep
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize weights and biases
        self.register_parameter('weight', nn.Parameter(torch.randn(visible_dim, hidden_dim) * 0.01))
        self.register_parameter('visible_bias', nn.Parameter(torch.zeros(visible_dim)))
        self.register_parameter('hidden_bias', nn.Parameter(torch.zeros(hidden_dim)))
        
        # Move to device
        self.to(self.device)
    
    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        Calculate the free energy of a visible state.
        
        Args:
            v: Visible state tensor [batch_size, visible_dim]
            
        Returns:
            Free energy of each sample in the batch
        """
        wx_b = F.linear(v, self.weight.t(), self.hidden_bias)
        hidden_term = torch.sum(torch.log(1 + torch.exp(wx_b)), dim=1)
        vbias_term = torch.mv(v, self.visible_bias)
        return -hidden_term - vbias_term
    
    def sample_h_given_v(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hidden state given visible state.
        
        Args:
            v: Visible state tensor [batch_size, visible_dim]
            
        Returns:
            Tuple of (hidden_probabilities, hidden_samples)
        """
        h_prob = torch.sigmoid(F.linear(v, self.weight.t(), self.hidden_bias))
        h_samples = torch.bernoulli(h_prob)
        return h_prob, h_samples
    
    def sample_v_given_h(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample visible state given hidden state.
        
        Args:
            h: Hidden state tensor [batch_size, hidden_dim]
            
        Returns:
            Tuple of (visible_probabilities, visible_samples)
        """
        v_prob = torch.sigmoid(F.linear(h, self.weight, self.visible_bias))
        v_samples = torch.bernoulli(v_prob)
        return v_prob, v_samples
    
    def forward(self, v: torch.Tensor, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform k steps of Gibbs sampling.
        
        Args:
            v: Visible state tensor [batch_size, visible_dim]
            k: Number of Gibbs steps
            
        Returns:
            Tuple of (visible_probabilities, visible_samples, hidden_probabilities)
        """
        h_prob, h_samples = self.sample_h_given_v(v)
        
        # Perform k steps of Gibbs sampling
        for _ in range(k):
            v_prob, v_samples = self.sample_v_given_h(h_samples)
            h_prob, h_samples = self.sample_h_given_v(v_samples)
        
        return v_prob, v_samples, h_prob
    
    def get_reconstruction_loss(self, v: torch.Tensor) -> torch.Tensor:
        """
        Calculate reconstruction loss for a batch of visible states.
        
        Args:
            v: Visible state tensor [batch_size, visible_dim]
            
        Returns:
            Mean squared error between input and reconstruction
        """
        v_prob, _, _ = self.forward(v)
        return F.mse_loss(v_prob, v)

class CrossLevelDBN(nn.Module):
    """
    Cross-Level Deep Belief Network that conditions the denoising of one level on coarser levels.
    
    This extends the standard DBN with explicit cross-level connections to model
    dependencies between hierarchical levels.
    """
    def __init__(
        self,
        target_visible_dim: int,
        target_hidden_dim: int,
        source_visible_dims: List[int],
        timestep: int = 0,
        cross_level_factor: float = 0.5,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the Cross-Level DBN.
        
        Args:
            target_visible_dim: Dimension of target level visible (input) layer
            target_hidden_dim: Dimension of target level hidden layer
            source_visible_dims: Dimensions of source (coarser) levels visible layers
            timestep: Which timestep in the diffusion process this DBN models
            cross_level_factor: Factor controlling the strength of cross-level connections
            device: Device to place model on
        """
        super().__init__()
        self.target_visible_dim = target_visible_dim
        self.target_hidden_dim = target_hidden_dim
        self.source_visible_dims = source_visible_dims
        self.timestep = timestep
        self.cross_level_factor = cross_level_factor
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Number of source levels
        self.num_source_levels = len(source_visible_dims)
        
        # Initialize regular DBN weights and biases for target level
        self.register_parameter('weight', nn.Parameter(torch.randn(target_visible_dim, target_hidden_dim) * 0.01))
        self.register_parameter('visible_bias', nn.Parameter(torch.zeros(target_visible_dim)))
        self.register_parameter('hidden_bias', nn.Parameter(torch.zeros(target_hidden_dim)))
        
        # Initialize cross-level connections
        # Each source level connects to both visible and hidden units of the target level
        self.cross_level_weights = nn.ParameterList([
            nn.Parameter(torch.randn(source_dim, target_visible_dim) * 0.01 * cross_level_factor)
            for source_dim in source_visible_dims
        ])
        
        # Cross-connections to hidden units
        self.cross_hidden_weights = nn.ParameterList([
            nn.Parameter(torch.randn(source_dim, target_hidden_dim) * 0.01 * cross_level_factor)
            for source_dim in source_visible_dims
        ])
        
        # Move to device
        self.to(self.device)
    
    def free_energy(self, v: torch.Tensor, source_v: List[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate the free energy of a visible state with cross-level conditioning.
        
        Args:
            v: Target visible state tensor [batch_size, target_visible_dim]
            source_v: List of source visible states from coarser levels
            
        Returns:
            Free energy of each sample in the batch
        """
        # Calculate base DBN free energy
        wx_b = F.linear(v, self.weight.t(), self.hidden_bias)
        
        # Add cross-level influence to hidden activation if source states are provided
        if source_v is not None:
            for i, s_v in enumerate(source_v):
                # Influence from this source level to hidden units
                wx_b += F.linear(s_v, self.cross_hidden_weights[i].t())
        
        # Complete free energy calculation
        hidden_term = torch.sum(torch.log(1 + torch.exp(wx_b)), dim=1)
        vbias_term = torch.mv(v, self.visible_bias)
        
        # Add cross-level visible biases if source states are provided
        cross_visible_term = 0
        if source_v is not None:
            for i, s_v in enumerate(source_v):
                cross_visible_term += torch.sum(s_v @ self.cross_level_weights[i] * v, dim=1)
        
        return -hidden_term - vbias_term - cross_visible_term
    
    def sample_h_given_v(self, v: torch.Tensor, source_v: List[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hidden state given visible state with cross-level conditioning.
        
        Args:
            v: Target visible state tensor [batch_size, target_visible_dim]
            source_v: List of source visible states from coarser levels
            
        Returns:
            Tuple of (hidden_probabilities, hidden_samples)
        """
        # Base hidden activation
        h_act = F.linear(v, self.weight.t(), self.hidden_bias)
        
        # Add cross-level influence if source states are provided
        if source_v is not None:
            for i, s_v in enumerate(source_v):
                h_act += F.linear(s_v, self.cross_hidden_weights[i].t())
        
        h_prob = torch.sigmoid(h_act)
        h_samples = torch.bernoulli(h_prob)
        return h_prob, h_samples
    
    def sample_v_given_h(self, h: torch.Tensor, source_v: List[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample visible state given hidden state with cross-level conditioning.
        
        Args:
            h: Hidden state tensor [batch_size, target_hidden_dim]
            source_v: List of source visible states from coarser levels
            
        Returns:
            Tuple of (visible_probabilities, visible_samples)
        """
        # Base visible activation
        v_act = F.linear(h, self.weight, self.visible_bias)
        
        # Add cross-level influence if source states are provided
        if source_v is not None:
            for i, s_v in enumerate(source_v):
                v_act += F.linear(s_v, self.cross_level_weights[i])
        
        v_prob = torch.sigmoid(v_act)
        v_samples = torch.bernoulli(v_prob)
        return v_prob, v_samples
    
    def forward(self, v: torch.Tensor, source_latents: List[torch.Tensor] = None, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform k steps of Gibbs sampling with cross-level conditioning.
        
        Args:
            v: Target visible state tensor [batch_size, target_visible_dim]
            source_latents: List of source visible states from coarser levels
            k: Number of Gibbs steps
            
        Returns:
            Tuple of (visible_probabilities, visible_samples, hidden_probabilities)
        """
        # Prepare source visible states by flattening if needed
        source_v = None
        if source_latents is not None:
            source_v = [s.reshape(s.shape[0], -1) for s in source_latents]
        
        # Flatten input if needed
        if len(v.shape) > 2:
            v = v.reshape(v.shape[0], -1)
        
        # Initial hidden state sampling
        h_prob, h_samples = self.sample_h_given_v(v, source_v)
        
        # Perform k steps of Gibbs sampling
        for _ in range(k):
            v_prob, v_samples = self.sample_v_given_h(h_samples, source_v)
            h_prob, h_samples = self.sample_h_given_v(v_samples, source_v)
        
        return v_prob, v_samples, h_prob
    
    def get_reconstruction_loss(self, v: torch.Tensor, source_latents: List[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate reconstruction loss for a batch of visible states with cross-level conditioning.
        
        Args:
            v: Target visible state tensor [batch_size, target_visible_dim]
            source_latents: List of source visible states from coarser levels
            
        Returns:
            Mean squared error between input and reconstruction
        """
        # Flatten input if needed
        if len(v.shape) > 2:
            v_flat = v.reshape(v.shape[0], -1)
        else:
            v_flat = v
        
        # Prepare source visible states by flattening if needed
        source_v = None
        if source_latents is not None:
            source_v = [s.reshape(s.shape[0], -1) for s in source_latents]
        
        v_prob, _, _ = self.forward(v_flat, source_v)
        return F.mse_loss(v_prob, v_flat)

class DiffusionScheduler(nn.Module):
    """
    Manages the diffusion process schedule with enhanced level-specific and cross-level conditioning.
    """
    def __init__(
        self,
        num_timesteps: int = 1000,
        num_levels: int = 3,
        schedule_types: Union[str, List[str]] = 'cosine',
        beta_start: Union[float, List[float]] = 0.0001,
        beta_end: Union[float, List[float]] = 0.02,
        cross_level_conditioning: bool = True
    ):
        """
        Initialize the diffusion scheduler with level-specific parameters.
        
        Args:
            num_timesteps: Number of diffusion timesteps
            num_levels: Number of hierarchical levels
            schedule_types: Type of noise schedule ('linear', 'cosine', 'quadratic')
                            Can be a single string for all levels or a list for level-specific schedules
            beta_start: Starting noise level(s)
                        Can be a single float for all levels or a list for level-specific values
            beta_end: Ending noise level(s)
                     Can be a single float for all levels or a list for level-specific values
            cross_level_conditioning: Whether to use cross-level conditioning in the diffusion process
        """
        super().__init__()
        self.num_timesteps = num_timesteps
        self.num_levels = num_levels
        self.cross_level_conditioning = cross_level_conditioning
        
        # Handle level-specific or shared schedule types
        if isinstance(schedule_types, str):
            self.schedule_types = [schedule_types] * num_levels
        else:
            assert len(schedule_types) == num_levels, "Must provide schedule_types for each level"
            self.schedule_types = schedule_types
        
        # Handle level-specific or shared beta values
        if isinstance(beta_start, float):
            self.beta_start = [beta_start] * num_levels
        else:
            assert len(beta_start) == num_levels, "Must provide beta_start for each level"
            self.beta_start = beta_start
            
        if isinstance(beta_end, float):
            self.beta_end = [beta_end] * num_levels
        else:
            assert len(beta_end) == num_levels, "Must provide beta_end for each level"
            self.beta_end = beta_end
        
        # Initialize betas, alphas, etc. for each level
        self.betas = []
        self.alphas = []
        self.alphas_cumprod = []
        self.alphas_cumprod_prev = []
        self.sqrt_alphas_cumprod = []
        self.sqrt_one_minus_alphas_cumprod = []
        self.sqrt_recip_alphas = []
        self.posterior_variance = []
        
        # Cross-level influence factors (learned if cross_level_conditioning is True)
        if cross_level_conditioning:
            # Parameters for cross-level influence (for each target level, from each source level)
            # We only condition on coarser levels, so we create a lower triangular matrix
            self.register_buffer('cross_level_factors', torch.zeros(num_levels, num_levels))
            
            # Initialize with reasonable defaults: coarser levels influence finer ones
            for target_level in range(1, num_levels):
                for source_level in range(target_level):
                    # More influence from immediately coarser level, less from much coarser levels
                    self.cross_level_factors[target_level, source_level] = 0.2 / (target_level - source_level)
        
        # Set up all schedules
        self._setup_schedules()
    
    def _setup_schedules(self):
        """Set up noise schedules and derived values for each level."""
        for level in range(self.num_levels):
            # Create the noise schedule based on the specified type
            if self.schedule_types[level] == 'linear':
                betas = torch.linspace(self.beta_start[level], self.beta_end[level], self.num_timesteps)
            elif self.schedule_types[level] == 'cosine':
                steps = torch.arange(self.num_timesteps + 1, dtype=torch.float32) / self.num_timesteps
                alpha_cumprod = torch.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
                alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
                betas = 1 - alpha_cumprod[1:] / alpha_cumprod[:-1]
                betas = torch.clamp(betas, 0.0001, 0.9999)
            elif self.schedule_types[level] == 'quadratic':
                betas = torch.linspace(self.beta_start[level]**0.5, self.beta_end[level]**0.5, self.num_timesteps) ** 2
            else:
                raise ValueError(f"Unknown schedule type: {self.schedule_types[level]}")
            
            # Store the betas for this level
            self.betas.append(betas)
            
            # Calculate derived values
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
            sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
            sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
            sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
            posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            
            # Store all derived values
            self.alphas.append(alphas)
            self.alphas_cumprod.append(alphas_cumprod)
            self.alphas_cumprod_prev.append(alphas_cumprod_prev)
            self.sqrt_alphas_cumprod.append(sqrt_alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod.append(sqrt_one_minus_alphas_cumprod)
            self.sqrt_recip_alphas.append(sqrt_recip_alphas)
            self.posterior_variance.append(posterior_variance)
    
    def _cross_level_influence(self, latents: List[torch.Tensor], target_level: int, t: torch.Tensor) -> torch.Tensor:
        """
        Calculate the cross-level influence for a target level.
        
        Args:
            latents: List of latent tensors from all levels
            target_level: Index of the target level to compute influence for
            t: Timestep tensor [batch_size]
            
        Returns:
            Tensor with cross-level influence values
        """
        if not self.cross_level_conditioning or target_level == 0:
            # No cross-level conditioning or this is the coarsest level (no conditioning from coarser levels)
            batch_size = latents[target_level].shape[0]
            return torch.zeros_like(latents[target_level])
            
        # Initialize influence tensor
        influence = torch.zeros_like(latents[target_level])
        
        # Combine influence from all coarser levels
        for source_level in range(target_level):
            # Skip if influence factor is zero
            if self.cross_level_factors[target_level, source_level] == 0:
                continue
                
            # Get influence factor for this level pair
            factor = self.cross_level_factors[target_level, source_level]
            
            # Get source latent and resize to match target shape if needed
            source_latent = latents[source_level]
            if source_latent.shape[2:] != latents[target_level].shape[2:]:
                # Resize spatial dimensions using adaptive pooling or interpolation
                source_latent = F.interpolate(
                    source_latent, 
                    size=latents[target_level].shape[2:],
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Channel dimension may also differ, we'll use a projection if needed
            if source_latent.shape[1] != latents[target_level].shape[1]:
                # Use 1x1 convolution to match channels (this is a simplified version)
                # In practice, this would be a learned projection matrix
                source_latent = source_latent.mean(dim=1, keepdim=True).expand(-1, latents[target_level].shape[1], -1, -1)
            
            # Add weighted influence
            influence += factor * source_latent
        
        return influence
    
    def add_noise(self, latents: List[torch.Tensor], t: torch.Tensor) -> List[torch.Tensor]:
        """
        Add noise to the input according to the level-specific diffusion schedules
        with cross-level conditioning.
        
        Args:
            latents: List of latent tensors [batch_size, channels, height, width] for each level
            t: Timestep tensor [batch_size]
            
        Returns:
            List of noisy tensors with same shapes as inputs
        """
        noisy_latents = []
        
        for level, latent in enumerate(latents):
            # Generate noise for this level
            noise = torch.rand_like(latent)
            
            # Get level-specific noise parameters
            sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod[level], t, latent.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod[level], t, latent.shape)
            
            # Apply cross-level influence if enabled
            if self.cross_level_conditioning and level > 0:
                # Calculate cross-level influence
                influence = self._cross_level_influence(latents, level, t)
                
                # Add noise with cross-level influence
                noisy_latent = sqrt_alphas_cumprod_t * latent + sqrt_one_minus_alphas_cumprod_t * (0.5 + influence) * noise
            else:
                # Standard noise addition
                noisy_latent = sqrt_alphas_cumprod_t * latent + sqrt_one_minus_alphas_cumprod_t * noise
            
            # Binarize the result for binary latent variables
            noisy_latent = (noisy_latent > 0.5).float()
            noisy_latents.append(noisy_latent)
        
        return noisy_latents
    
    def compute_denoise_weight(self, level: int, t_prev: int, t: int) -> float:
        """
        Compute the weight for denoising between timesteps for a specific level.
        
        Args:
            level: Hierarchy level index
            t_prev: Previous timestep
            t: Current timestep
            
        Returns:
            Denoising weight for this level and timestep pair
        """
        alpha_cumprod_t = self.alphas_cumprod[level][t]
        alpha_cumprod_t_prev = self.alphas_cumprod[level][t_prev]
        
        return (1.0 - alpha_cumprod_t_prev) / (1.0 - alpha_cumprod_t) * (1.0 - alpha_cumprod_t / alpha_cumprod_t_prev)

# Helper function for extracting elements from a tensor at indices
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu()).reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    return out

class DBNDenoiser(nn.Module):
    """
    DBN-based denoiser for binary diffusion models with enhanced cross-level conditioning.
    """
    def __init__(
        self,
        latent_sizes: List[int],
        hidden_units: Optional[List[int]] = None,
        hidden_dim_factor: float = 2.0,
        num_timesteps: int = 1000,
        dbns_per_run: int = 10,
        cross_level_conditioning: bool = True,
        schedule_types: Union[str, List[str]] = 'cosine',
        device: Optional[torch.device] = None
    ):
        """
        Initialize the DBN denoiser.
        
        Args:
            latent_sizes: List of binary latent sizes for each level
            hidden_units: Optional list of hidden units for each level's DBNs
            hidden_dim_factor: Factor to determine hidden dimension size relative to visible dimension
            num_timesteps: Number of timesteps in the diffusion process
            dbns_per_run: Number of DBNs to process in a single quantum annealing run
            cross_level_conditioning: Whether to use cross-level conditioning in the diffusion process
            schedule_types: Type(s) of noise schedule ('linear', 'cosine', 'quadratic')
            device: Device to place model on
        """
        super().__init__()
        self.latent_sizes = latent_sizes
        self.num_levels = len(latent_sizes)
        self.hidden_dim_factor = hidden_dim_factor
        self.num_timesteps = num_timesteps
        self.dbns_per_run = dbns_per_run
        self.cross_level_conditioning = cross_level_conditioning
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set up hidden units for each level if not provided
        if hidden_units is None:
            self.hidden_units = []
            for latent_size in latent_sizes:
                self.hidden_units.append(int(latent_size * hidden_dim_factor))
        else:
            assert len(hidden_units) == len(latent_sizes), "Must provide same number of hidden_units as latent_sizes"
            self.hidden_units = hidden_units
        
        # Create cross-level connections for conditioning
        if cross_level_conditioning:
            # Cross-level connection modules
            self.cross_level_connectors = nn.ModuleList([
                nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(latent_sizes[source_level], latent_sizes[target_level]),
                        nn.Sigmoid()
                    ) if source_level < target_level else None
                    for source_level in range(self.num_levels)
                ])
                for target_level in range(self.num_levels)
            ])
        
        # Create DBNs for each timestep and level with cross-level conditioning
        self.dbns = nn.ModuleList([])
        
        for t in range(num_timesteps):
            level_dbns = nn.ModuleList([])
            
            for level_idx, latent_size in enumerate(latent_sizes):
                # Base visible dimension is the latent size for this level
                visible_dim = latent_size
                
                # If using cross-level conditioning, augment visible dimension for coarser level states
                if cross_level_conditioning and level_idx > 0:
                    # We add a cross-level influence vector to the visible state
                    cl_dbn = CrossLevelDBN(
                        target_visible_dim=latent_size,
                        target_hidden_dim=self.hidden_units[level_idx],
                        source_visible_dims=[latent_sizes[l] for l in range(level_idx)],
                        timestep=t,
                        device=self.device
                    )
                    level_dbns.append(cl_dbn)
                else:
                    # Standard DBN without cross-level conditioning
                    dbn = DeepBeliefNetwork(
                        visible_dim=visible_dim,
                        hidden_dim=self.hidden_units[level_idx],
                        timestep=t,
                        device=self.device
                    )
                    level_dbns.append(dbn)
            
            self.dbns.append(level_dbns)
        
        # Set up diffusion scheduler with cross-level conditioning
        self.scheduler = DiffusionScheduler(
            num_timesteps=num_timesteps,
            num_levels=self.num_levels,
            schedule_types=schedule_types,
            cross_level_conditioning=cross_level_conditioning
        )
    
    def denoise_step(
        self,
        latents: List[torch.Tensor],
        t: int,
        quantum_sampler=None,
        sampling_steps: int = 1
    ) -> List[torch.Tensor]:
        """
        Perform a single denoising step on the latent representation with cross-level conditioning.
        
        Args:
            latents: List of binary latent tensors to denoise
            t: Current timestep
            quantum_sampler: Optional quantum sampler for better sampling
            sampling_steps: Number of Gibbs sampling steps (if not using quantum)
            
        Returns:
            List of denoised binary latent tensors
        """
        # Check if we're at the last timestep
        if t <= 0:
            return latents
        
        # Create list for denoised latents
        denoised_latents = []
        
        # Process each level separately, from coarsest to finest, to allow cross-level conditioning
        for level_idx, latent in enumerate(latents):
            # Get the DBN for this timestep and level
            dbn = self.dbns[t][level_idx]
            
            # Check if using cross-level conditioning
            if self.cross_level_conditioning and level_idx > 0 and isinstance(dbn, CrossLevelDBN):
                # Prepare source latents from coarser levels for conditioning
                source_latents = [latents[l] for l in range(level_idx)]
                
                # Check if we should use quantum sampling
                if quantum_sampler is not None:
                    # Use quantum sampling for a DBN with cross-level conditioning
                    _, denoised, _ = quantum_sampler.sample_cl_dbn(
                        dbn, 
                        latent, 
                        source_latents=source_latents
                    )
                else:
                    # Use standard Gibbs sampling with cross-level conditioning
                    _, denoised, _ = dbn(latent, source_latents=source_latents, k=sampling_steps)
            else:
                # Standard denoising without cross-level conditioning
                # Check if we should use quantum sampling
                if quantum_sampler is not None:
                    # Check if quantum sampler has multi_anneal capability
                    use_multi_anneal = hasattr(quantum_sampler, 'multi_anneal') and quantum_sampler.multi_anneal
                    
                    # Check if we can batch multiple DBNs at once
                    if t >= self.dbns_per_run:
                        # Determine how many timesteps to batch
                        num_batch_timesteps = min(self.dbns_per_run, t + 1)
                        
                        # Gather the DBNs for this batch
                        dbns_batch = [self.dbns[t - i][level_idx] for i in range(num_batch_timesteps)]
                        
                        # Create inputs for each DBN (all are the same current latent)
                        inputs_batch = [latent for _ in range(num_batch_timesteps)]
                        
                        # Use quantum sampler for batch sampling
                        batch_results = quantum_sampler.sample_dbns_batch(dbns_batch, inputs_batch)
                        
                        # Use the first result (corresponding to current timestep)
                        denoised = batch_results[0]
                    else:
                        # Use quantum sampling for a single DBN
                        if use_multi_anneal:
                            # Get probabilities and samples with multi-anneal approach
                            _, denoised, _ = quantum_sampler._sample_dbn_multi_anneal(dbn, latent)
                        else:
                            # Standard quantum sampling
                            _, denoised, _ = quantum_sampler.sample_dbn(dbn, latent)
                else:
                    # Use standard Gibbs sampling
                    _, denoised, _ = dbn(latent, k=sampling_steps)
            
            denoised_latents.append(denoised)
        
        return denoised_latents
    
    def forward(
        self,
        latents: List[torch.Tensor],
        noise_level: Optional[float] = None,
        steps: Optional[int] = None,
        quantum_sampler=None
    ) -> List[torch.Tensor]:
        """
        Denoise the latent representation with cross-level conditioning.
        
        Args:
            latents: List of binary latent tensors to denoise
            noise_level: Optional noise level (0 to 1) to add and then denoise
            steps: Number of denoising steps (defaults to all)
            quantum_sampler: Optional quantum sampler for better sampling
            
        Returns:
            List of denoised binary latent tensors
        """
        # Handle the case where we want to add noise and then denoise
        if noise_level is not None:
            # Scale noise_level to timestep
            t_start = min(int(noise_level * self.num_timesteps), self.num_timesteps - 1)
            
            # Add noise to all latent levels using the cross-level scheduler
            batch_size = latents[0].shape[0]
            t_batch = torch.full((batch_size,), t_start, device=self.device, dtype=torch.long)
            noisy_latents = self.scheduler.add_noise(latents, t_batch)
            
            # Set latents to the noisy version for denoising
            latents = noisy_latents
        else:
            # If no noise level specified, assume latents are already noisy at max noise
            t_start = self.num_timesteps - 1
        
        # Determine number of denoising steps
        if steps is None:
            # Default to full denoising
            steps = t_start + 1
        else:
            # Limit steps to available timesteps
            steps = min(steps, t_start + 1)
        
        # Check if we have a quantum sampler with multi-anneal capability
        use_multi_anneal = quantum_sampler is not None and hasattr(quantum_sampler, 'multi_anneal') and quantum_sampler.multi_anneal
        
        # Denoise progressively
        current_latents = latents
        
        # If using multi-anneal, keep track of the best solutions across steps
        best_latents = current_latents
        best_energy = None
        
        for t in range(t_start, t_start - steps, -1):
            # Perform one denoising step
            current_latents = self.denoise_step(
                current_latents, 
                t, 
                quantum_sampler=quantum_sampler,
                sampling_steps=1 if quantum_sampler else 10  # More Gibbs steps if no quantum sampler
            )
            
            # If using multi-anneal, evaluate and potentially update best solutions
            if use_multi_anneal and quantum_sampler is not None:
                # For simplicity, we'll use the next timestep's DBNs to evaluate energy
                # This is an approximation but works well in practice
                next_t = max(0, t - 1)
                
                # Calculate energy for current solution (lower is better)
                current_energy = 0
                for level_idx, latent in enumerate(current_latents):
                    dbn = self.dbns[next_t][level_idx]
                    # Simple energy approximation: free energy
                    level_energy = dbn.free_energy(latent.view(latent.size(0), -1)).mean()
                    current_energy += level_energy
                
                # Update best solutions if this is the first evaluation or better than previous best
                if best_energy is None or current_energy < best_energy:
                    best_latents = [latent.clone() for latent in current_latents]
                    best_energy = current_energy
        
        # Return best solutions if we tracked them, otherwise current latents
        if use_multi_anneal and best_energy is not None:
            return best_latents
        else:
            return current_latents
    
    def train_with_quantum(
        self,
        train_data: List[List[torch.Tensor]],
        quantum_sampler=None,
        epochs: int = 10,
        cd_k: int = 1,
        lr: float = 0.01,
        momentum: float = 0.9,
        batch_size: int = 32
    ) -> Dict[str, List[float]]:
        """
        Train the DBNs with optional quantum accelerated sampling.
        
        Args:
            train_data: List of Lists of binary latent tensors (timesteps × levels × samples)
            quantum_sampler: Optional quantum sampler for better MCMC
            epochs: Number of training epochs
            cd_k: Number of contrastive divergence steps
            lr: Learning rate
            momentum: Momentum for SGD
            batch_size: Batch size for training
            
        Returns:
            Dictionary of training metrics
        """
        # Check that we have training data for each timestep
        assert len(train_data) == self.num_timesteps, "Need training data for each timestep"
        
        # Prepare optimizers for each DBN
        optimizers = [[
            torch.optim.SGD(self.dbns[t][level].parameters(), lr=lr, momentum=momentum)
            for level in range(self.num_levels)
        ] for t in range(self.num_timesteps)]
        
        # Training metrics
        metrics = {
            "epoch_losses": [],
            "level_losses": [[] for _ in range(self.num_levels)],
            "timestep_losses": [[] for _ in range(self.num_timesteps)]
        }
        
        # Training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            
            # Process each timestep
            for t in range(self.num_timesteps):
                # Check that we have the right number of levels
                assert len(train_data[t]) == self.num_levels, f"Need data for each level at timestep {t}"
                
                # Process each level
                for level in range(self.num_levels):
                    # Get data for this level and timestep
                    level_data = train_data[t][level]
                    
                    # Skip if no data
                    if level_data is None or len(level_data) == 0:
                        continue
                    
                    # Convert to tensor if not already
                    if not isinstance(level_data, torch.Tensor):
                        level_data = torch.tensor(level_data, device=self.device)
                    
                    # Make sure it's on the right device
                    level_data = level_data.to(self.device)
                    
                    # Get the DBN for this timestep and level
                    dbn = self.dbns[t][level]
                    optimizer = optimizers[t][level]
                    
                    # Calculate loss for the epoch
                    total_loss = 0.0
                    num_batches = 0
                    
                    # Process in batches
                    for i in range(0, len(level_data), batch_size):
                        # Get batch
                        batch = level_data[i:i+batch_size]
                        if len(batch) == 0:
                            continue
                        
                        # Zero gradients
                        optimizer.zero_grad()
                        
                        # Perform CD-k
                        if quantum_sampler is not None and t % self.dbns_per_run == 0:
                            # Check if quantum sampler has multi_anneal capability
                            use_multi_anneal = hasattr(quantum_sampler, 'multi_anneal') and quantum_sampler.multi_anneal
                            
                            # Batch multiple DBNs for the quantum annealer
                            dbns_to_batch = []
                            timesteps_to_batch = []
                            
                            # Collect DBNs for batching (this and next dbns_per_run-1 timesteps)
                            for batch_t in range(t, min(t + self.dbns_per_run, self.num_timesteps)):
                                dbns_to_batch.append(self.dbns[batch_t][level])
                                timesteps_to_batch.append(batch_t)
                            
                            # Create batch input (same for all DBNs in batch)
                            batch_inputs = [batch for _ in range(len(dbns_to_batch))]
                            
                            # Use quantum sampling for the batch
                            batch_results = quantum_sampler.sample_dbns_batch(dbns_to_batch, batch_inputs)
                            
                            # Get current timestep's result
                            v_samples = batch_results[0]
                            
                            # Compute probabilities
                            with torch.no_grad():
                                # Get hidden probabilities from visible samples
                                h_prob, _ = dbn.sample_h_given_v(v_samples)
                                
                                # Get visible probabilities from hidden probabilities
                                v_prob, _ = dbn.sample_v_given_h(h_prob)
                        else:
                            # Standard contrastive divergence
                            v_prob, v_samples, h_prob = dbn(batch, k=cd_k)
                        
                        # Calculate free energy difference (CD loss)
                        loss = torch.mean(dbn.free_energy(batch) - dbn.free_energy(v_samples))
                        
                        # Backward pass
                        loss.backward()
                        
                        # Update weights
                        optimizer.step()
                        
                        # Accumulate loss
                        total_loss += loss.item()
                        num_batches += 1
                    
                    # Calculate average loss for this level and timestep
                    if num_batches > 0:
                        avg_loss = total_loss / num_batches
                        metrics["level_losses"][level].append(avg_loss)
                        metrics["timestep_losses"][t].append(avg_loss)
                        epoch_loss += avg_loss
            
            # Calculate average loss for this epoch
            epoch_loss /= (self.num_timesteps * self.num_levels)
            metrics["epoch_losses"].append(epoch_loss)
            
            # Print progress
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}, Time: {epoch_time:.2f}s")
        
        return metrics 