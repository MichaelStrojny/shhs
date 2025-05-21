"""
Hierarchical Encoder

This module implements a hierarchical encoder that transforms input data
into a multi-level binary latent representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict, Optional

class ResidualBlock(nn.Module):
    """
    Residual block with optional downsampling for feature extraction.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class AttentionBlock(nn.Module):
    """
    Self-attention block for capturing long-range dependencies.
    """
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm = nn.LayerNorm([channels])
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(b, h*w, c)  # [B, H*W, C]
        
        # Apply layer norm
        x_norm = self.norm(x_flat)
        
        # Compute QKV
        qkv = self.qkv(x_norm).reshape(b, h*w, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, H*W, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(b, h*w, c)
        out = self.proj(out)
        
        # Add residual connection
        out = out + x_flat
        
        # Reshape back to spatial
        out = out.reshape(b, h, w, c).permute(0, 3, 1, 2)
        
        return out

class HierarchicalLevel(nn.Module):
    """
    Single level of the hierarchical encoder.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        downscale_factor: int = 2,
        use_attention: bool = True
    ):
        super().__init__()
        self.downscale_factor = downscale_factor
        
        # Feature extraction
        modules = []
        current_channels = in_channels
        
        # Initial convolution
        modules.append(nn.Conv2d(current_channels, hidden_channels, kernel_size=3, padding=1))
        modules.append(nn.BatchNorm2d(hidden_channels))
        modules.append(nn.ReLU(inplace=True))
        current_channels = hidden_channels
        
        # Residual blocks with downsampling
        for _ in range(int(math.log2(downscale_factor))):
            modules.append(ResidualBlock(current_channels, current_channels, stride=2))
        
        # Add attention block if requested
        if use_attention:
            modules.append(AttentionBlock(current_channels))
        
        # Final block for feature processing
        modules.append(ResidualBlock(current_channels, current_channels))
        
        # Final layer to produce output channels with sigmoid
        modules.append(nn.Conv2d(current_channels, out_channels, kernel_size=1))
        modules.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*modules)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process input and return continuous features in [0,1] range
        return self.model(x)

class StraightThroughBinarizer(torch.autograd.Function):
    """
    Enhanced straight-through estimator for binary values with optimal gradient flow.
    
    Forward: Hard threshold at 0.5
    Backward: Optimized gradient estimation with temperature-controlled scaling
    and proximity-based scaling for better training stability and convergence
    
    Based on binary.tex paper insights that indicate simple straight-through gradient
    estimation is sufficient for high-quality image reconstruction while maintaining
    high training efficiency.
    """
    @staticmethod
    def forward(ctx, input_tensor, temperature=1.0):
        ctx.save_for_backward(input_tensor)
        ctx.temperature = temperature
        return (input_tensor > 0.5).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        temperature = ctx.temperature
        
        # Calculate optimal gradient scaling factor based on proximity to decision boundary
        # This creates a gradient "window" allowing better flow near the decision boundary
        distance_to_boundary = torch.abs(input_tensor - 0.5)
        
        # Enhanced straight-through gradient with importance weighting
        # The key insight from binary.tex is that simple straight-through works well,
        # but we can improve it with focused gradients near the decision boundary
        gradient_scale = torch.ones_like(input_tensor)
        
        # Only modify gradient scale for values close to the decision boundary
        near_boundary = distance_to_boundary < 0.2
        if near_boundary.any():
            # Enhanced gradients near boundary
            gradient_scale[near_boundary] = 1.2 - distance_to_boundary[near_boundary] * 2.0
        
        # Adjust scaling based on temperature (lower temp = more focused gradients)
        if temperature < 1.0:
            gradient_scale = gradient_scale * (2.0 - temperature)
        
        # Return scaled gradients
        return grad_output * gradient_scale, None

class HierarchicalEncoder(nn.Module):
    """
    Enhanced hierarchical encoder with learned structure and adaptive bit allocation.
    Transforms input data into a multi-level binary latent representation.
    """
    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: int = 128,
        latent_dims: List[int] = [32, 16, 8],
        downscale_factors: List[int] = [8, 4, 2],
        max_levels: int = 4,
        min_levels: int = 2,
        use_attention: bool = True,
        spatial_dims: Tuple[int, int] = (32, 32),
        initial_temperature: float = 1.0,
        min_temperature: float = 0.5,
        anneal_rate: float = 0.9999,
        adaptive_hierarchy: bool = True,
        adaptive_bit_allocation: bool = True
    ):
        """
        Initialize the hierarchical encoder.
        
        Args:
            input_channels: Number of input channels
            hidden_channels: Number of hidden channels
            latent_dims: List of latent dimensions for each level
            downscale_factors: List of spatial downscaling factors for each level
            max_levels: Maximum number of hierarchical levels (for adaptive hierarchy)
            min_levels: Minimum number of hierarchical levels (for adaptive hierarchy)
            use_attention: Whether to use attention blocks
            spatial_dims: Input spatial dimensions (height, width)
            initial_temperature: Initial temperature for binarization
            min_temperature: Minimum temperature for binarization
            anneal_rate: Temperature annealing rate
            adaptive_hierarchy: Whether to learn the optimal hierarchy structure
            adaptive_bit_allocation: Whether to use advanced content-aware bit allocation
        """
        super().__init__()
        assert len(latent_dims) == len(downscale_factors), "Must provide same number of latent dims and downscale factors"
        assert len(latent_dims) <= max_levels, "Number of latent dims exceeds max_levels"
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.base_latent_dims = latent_dims
        self.base_downscale_factors = downscale_factors
        self.num_base_levels = len(latent_dims)
        self.max_levels = max_levels
        self.min_levels = min_levels
        self.spatial_dims = spatial_dims
        self.adaptive_hierarchy = adaptive_hierarchy
        self.adaptive_bit_allocation = adaptive_bit_allocation
        
        # Temperature parameters for binarization
        self.register_buffer('temperature', torch.tensor([initial_temperature]))
        self.min_temperature = min_temperature
        self.anneal_rate = anneal_rate
        
        # Prepare for potential additional adaptive levels
        extended_latent_dims = latent_dims.copy()
        extended_downscale_factors = downscale_factors.copy()
        
        # Add potential additional levels if using adaptive hierarchy
        if adaptive_hierarchy and len(latent_dims) < max_levels:
            # For additional levels, use half the channels of previous level
            # and double the downscale factor
            for i in range(len(latent_dims), max_levels):
                new_latent_dim = max(4, extended_latent_dims[-1] // 2)
                new_downscale = extended_downscale_factors[-1] * 2
                extended_latent_dims.append(new_latent_dim)
                extended_downscale_factors.append(new_downscale)
        
        # Store extended dimensions
        self.extended_latent_dims = extended_latent_dims
        self.extended_downscale_factors = extended_downscale_factors
        
        # Create learnable importance weights for each level
        if adaptive_hierarchy:
            # Create learnable importance weights to determine which levels to use
            self.level_importance = nn.Parameter(torch.ones(max_levels))
            
            # Initialize so that base levels have higher importance
            with torch.no_grad():
                # Base levels start with higher importance
                self.level_importance[:len(latent_dims)] = 2.0
                # Additional levels start with lower importance
                if len(latent_dims) < max_levels:
                    self.level_importance[len(latent_dims):] = 0.5
        
        # Create bit importance weighting for each latent dimension and level
        self.bit_importance = nn.ParameterList([
            nn.Parameter(torch.ones(dim))
            for dim in extended_latent_dims
        ])
        
        # Create encoder levels for all potential levels
        self.levels = nn.ModuleList()
        for i, (latent_dim, downscale) in enumerate(zip(extended_latent_dims, extended_downscale_factors)):
            self.levels.append(
                HierarchicalLevel(
                    in_channels=input_channels,
                    hidden_channels=hidden_channels,
                    out_channels=latent_dim,
                    downscale_factor=downscale,
                    use_attention=use_attention
                )
            )
        
        # Binarizer with straight-through estimator
        self.binarizer = StraightThroughBinarizer.apply
        
        # Enhanced spatial structure processing for better binary spatial correlations
        self.spatial_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1, groups=latent_dim),
                nn.BatchNorm2d(latent_dim),
                nn.SiLU(),
                nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
                nn.Sigmoid()
            )
            for latent_dim in extended_latent_dims
        ])
        
        # Create content complexity estimators for adaptive bit allocation
        if adaptive_bit_allocation:
            self.complexity_estimators = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(latent_dim, latent_dim // 2, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(latent_dim // 2, 1, kernel_size=1),
                    nn.Sigmoid()
                )
                for latent_dim in extended_latent_dims
            ])
        
        # Level selection gating for adaptive hierarchy
        if adaptive_hierarchy:
            # Module that helps determine optimal number of levels based on input complexity
            self.level_selector = nn.Sequential(
                nn.AdaptiveAvgPool2d(8),  # Reduce spatial dimensions for global analysis
                nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(hidden_channels * 8 * 8, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, max_levels),
                nn.Sigmoid()  # Output per-level importance between 0 and 1
            )
            
            # Cross-level attention modules for better information flow
            self.cross_level_attention = nn.ModuleList([
                nn.ModuleList([
                    # Attention from source level to target level (if source < target)
                    nn.Sequential(
                        nn.Conv2d(extended_latent_dims[source_idx], extended_latent_dims[target_idx], kernel_size=1),
                        nn.Sigmoid()
                    ) if source_idx < target_idx else None
                    for source_idx in range(max_levels)
                ])
                for target_idx in range(max_levels)
            ])
    
    def _get_active_level_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute which levels to use for this input based on content complexity.
        
        Args:
            x: Input tensor [batch_size, input_channels, height, width]
            
        Returns:
            Tensor of level importance weights [batch_size, max_levels]
        """
        if not self.adaptive_hierarchy:
            # If not using adaptive hierarchy, just use base levels
            batch_size = x.shape[0]
            weights = torch.zeros(batch_size, self.max_levels, device=x.device)
            weights[:, :self.num_base_levels] = 1.0
            return weights
            
        # Use level selector to determine importance of each level
        input_complexity = self.level_selector(x)  # [batch_size, max_levels]
        
        # Combine with learned level importance
        level_weights = input_complexity * self.level_importance
        
        # Apply softmax to get relative importance
        importance = F.softmax(level_weights, dim=1)
        
        # Create binary level selection with straight-through estimator for training
        if self.training:
            # During training, we use Gumbel softmax to get differentiable discrete samples
            level_selection = F.gumbel_softmax(level_weights, tau=max(0.5, self.temperature.item()), hard=True)
            
            # Ensure we use at least min_levels and at most max_levels
            # We sort importance and select top-k levels
            _, top_indices = torch.topk(importance, k=min(self.max_levels, self.min_levels))
            mask = torch.zeros_like(importance)
            mask.scatter_(1, top_indices, 1.0)
            
            # Combine differentiable selection with mask to ensure min_levels
            level_selection = level_selection * (1 - mask) + mask
        else:
            # During inference, directly select top-k levels based on importance
            k = max(self.min_levels, (importance > 0.1).sum(dim=1).clamp(max=self.max_levels).item())
            _, top_indices = torch.topk(importance, k=min(k, self.max_levels))
            level_selection = torch.zeros_like(importance)
            level_selection.scatter_(1, top_indices, 1.0)
        
        return level_selection
    
    def compute_content_complexity(self, latents: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Compute content complexity maps for adaptive bit allocation.
        
        Args:
            latents: List of continuous latent tensors
            
        Returns:
            List of complexity maps, one per level
        """
        if not self.adaptive_bit_allocation:
            # Return uniform complexity (all ones) if not using adaptive bit allocation
            return [torch.ones_like(latent[:, :1, :, :]) for latent in latents]
        
        complexity_maps = []
        for i, latent in enumerate(latents):
            # Use complexity estimator to analyze this level's content
            complexity = self.complexity_estimators[i](latent)
            
            # Add local variation as another signal of complexity (gradient magnitude)
            local_var = torch.abs(F.avg_pool2d(latent, 3, stride=1, padding=1) - latent).mean(dim=1, keepdim=True)
            
            # Combine both signals
            combined_complexity = (complexity + local_var) / 2
            
            # Normalize to [0.25, 1.0] range to ensure minimal representation everywhere
            normalized = 0.25 + 0.75 * combined_complexity
            
            complexity_maps.append(normalized)
        
        return complexity_maps
    
    def apply_cross_level_attention(self, latents: List[torch.Tensor], level_weights: torch.Tensor) -> List[torch.Tensor]:
        """
        Apply cross-level attention to enhance information flow between levels.
        
        Args:
            latents: List of continuous latent tensors
            level_weights: Tensor of level importance weights [batch_size, max_levels]
            
        Returns:
            List of refined latent tensors
        """
        if not self.adaptive_hierarchy or len(latents) <= 1:
            return latents
            
        refined_latents = []
        num_levels = len(latents)
        
        for target_idx, target_latent in enumerate(latents):
            # Skip if this level isn't active
            if level_weights[0, target_idx] <= 0:
                refined_latents.append(target_latent)
                continue
                
            # Start with the original target latent
            refined = target_latent
            
            # Apply attention from all active source levels to this target
            for source_idx in range(num_levels):
                # Skip if source isn't active or we don't have attention for this pair
                if (source_idx == target_idx or 
                    level_weights[0, source_idx] <= 0 or 
                    source_idx >= target_idx or
                    self.cross_level_attention[target_idx][source_idx] is None):
                    continue
                
                # Get source latent and apply attention
                source_latent = latents[source_idx]
                
                # Adjust source spatial dimensions to match target if needed
                if source_latent.shape[2:] != target_latent.shape[2:]:
                    source_latent = F.interpolate(
                        source_latent,
                        size=target_latent.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Apply attention from source to target
                attention_weights = self.cross_level_attention[target_idx][source_idx](source_latent)
                refined = refined * (1 + attention_weights)
            
            # Add to list of refined latents
            refined_latents.append(refined)
        
        return refined_latents
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Encode input data into hierarchical binary latent representation with adaptive
        hierarchy and content-aware bit allocation.
        
        Args:
            x: Input tensor [batch_size, input_channels, height, width]
            
        Returns:
            List of binary latent tensors, one for each active level
        """
        # Determine which levels to use for this input
        level_weights = self._get_active_level_weights(x)
        active_level_indices = torch.where(level_weights[0] > 0)[0].tolist()
        
        # Ensure we have at least min_levels
        if len(active_level_indices) < self.min_levels:
            # Add most important inactive levels
            remaining = self.min_levels - len(active_level_indices)
            inactive_indices = torch.where(level_weights[0] <= 0)[0].tolist()
            inactive_importance = self.level_importance[inactive_indices]
            top_inactive = torch.topk(inactive_importance, k=min(remaining, len(inactive_indices)))[1]
            additional_indices = [inactive_indices[i.item()] for i in top_inactive]
            active_level_indices.extend(additional_indices)
        
        # Sort indices by original order
        active_level_indices.sort()
        
        # Process each active level to get continuous latent representations
        cont_latents = []
        for i in active_level_indices:
            # Get continuous latent representation
            cont_latent = self.levels[i](x)
            cont_latents.append(cont_latent)
        
        # Compute content complexity for adaptive bit allocation
        complexity_maps = self.compute_content_complexity(cont_latents)
        
        # Apply cross-level attention for better information flow
        refined_latents = self.apply_cross_level_attention(cont_latents, level_weights)
        
        # Process refined latents with spatial processors and binarization
        binary_latents = []
        for i, (level_idx, cont_latent, complexity) in enumerate(zip(active_level_indices, refined_latents, complexity_maps)):
            # Apply spatial processing to enhance structure awareness
            spatial_weight = self.spatial_processors[level_idx](cont_latent)
            cont_latent = cont_latent * spatial_weight
            
            # Apply content-aware bit allocation using complexity maps
            if self.adaptive_bit_allocation:
                # Expand complexity map to match channel dimensions
                expanded_complexity = complexity.expand(-1, cont_latent.shape[1], -1, -1)
                
                # Apply bit importance weighting from learned parameters
                bit_weights = self.bit_importance[level_idx].view(1, -1, 1, 1).expand_as(cont_latent)
                
                # Combine learned bit importance with content complexity
                cont_latent = cont_latent * bit_weights * expanded_complexity
            else:
                # Apply only learned bit importance if not using adaptive allocation
                bit_weights = self.bit_importance[level_idx].view(1, -1, 1, 1).expand_as(cont_latent)
                cont_latent = cont_latent * bit_weights
            
            # Binarize using enhanced straight-through estimator with temperature
            binary_latent = self.binarizer(cont_latent, self.temperature.item())
            
            # Append to list of latents
            binary_latents.append(binary_latent)
        
        # Update temperature if in training mode
        if self.training:
            with torch.no_grad():
                self.temperature.mul_(self.anneal_rate).clamp_(min=self.min_temperature)
        
        return binary_latents
    
    def calculate_latent_sizes(self) -> List[int]:
        """
        Calculate the number of binary units in each latent level.
        
        Returns:
            List of sizes (number of binary units) for each level
        """
        sizes = []
        
        if self.adaptive_hierarchy:
            # During initialization or when checking architecture, use all base levels
            for i, (latent_dim, downscale) in enumerate(zip(self.base_latent_dims, self.base_downscale_factors)):
                h = self.spatial_dims[0] // downscale
                w = self.spatial_dims[1] // downscale
                sizes.append(latent_dim * h * w)
        else:
            # Calculate standard static sizes
            for i, (latent_dim, downscale) in enumerate(zip(self.extended_latent_dims, self.extended_downscale_factors)):
                if i < self.num_base_levels:  # Only include base levels if not adaptive
                    h = self.spatial_dims[0] // downscale
                    w = self.spatial_dims[1] // downscale
                    sizes.append(latent_dim * h * w)
        
        return sizes
    
    def get_total_latent_size(self) -> int:
        """
        Calculate the total number of binary units across all levels.
        
        Returns:
            Total number of binary units
        """
        return sum(self.calculate_latent_sizes()) 