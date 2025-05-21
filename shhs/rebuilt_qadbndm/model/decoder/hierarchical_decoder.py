"""
Hierarchical Decoder

This module implements a hierarchical decoder that reconstructs data
from multi-level binary latent representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict, Optional

class ResidualBlock(nn.Module):
    """
    Residual block with optional upsampling for feature reconstruction.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class UpsampleBlock(nn.Module):
    """
    Upsampling block for feature reconstruction.
    """
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)

class AttentionBlock(nn.Module):
    """
    Self-attention block for capturing long-range dependencies in features.
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

class HierarchicalLevelDecoder(nn.Module):
    """
    Decoder for a single level of the hierarchical representation with enhanced
    binary representation handling based on binary.tex insights.
    """
    def __init__(
        self,
        latent_channels: int,
        hidden_channels: int,
        output_channels: int,
        upscale_factor: int = 2,
        use_attention: bool = True,
        channel_multiplier: int = 4,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.upscale_factor = upscale_factor
        expanded_channels = hidden_channels * channel_multiplier
        
        # Enhanced binary latent processing
        # The binary.tex paper showed binary features need special handling
        self.binary_processor = nn.Sequential(
            nn.Conv2d(latent_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(), # SiLU (Swish) activation better preserves binary information
            ResidualBlock(hidden_channels, hidden_channels)
        )
        
        # Apply attention if requested, with enhanced multi-head attention
        if use_attention:
            self.attention = nn.Sequential(
                AttentionBlock(hidden_channels, num_heads=8),
                ResidualBlock(hidden_channels, hidden_channels)
            )
        else:
            self.attention = nn.Identity()
        
        # Channel expansion for better feature processing
        self.expansion = nn.Sequential(
            nn.Conv2d(hidden_channels, expanded_channels, kernel_size=1),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(),
            nn.Dropout2d(dropout_rate)
        )
        
        # Upsampling blocks with skip connections
        self.upsampling_blocks = nn.ModuleList()
        current_channels = expanded_channels
        
        for i in range(int(math.log2(upscale_factor))):
            # Gradually reduce channels as we upsample
            next_channels = current_channels // 2 if i > 0 else current_channels
            self.upsampling_blocks.append(
                nn.Sequential(
                    UpsampleBlock(current_channels, next_channels),
                    ResidualBlock(next_channels, next_channels),
                )
            )
            current_channels = next_channels
        
        # Final processing with better perceptual quality
        self.final_processor = nn.Sequential(
            nn.Conv2d(current_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(),
            ResidualBlock(hidden_channels, hidden_channels),
            nn.Conv2d(hidden_channels, output_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Tanh for final output to match normed pixel space [-1, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode binary latent to reconstructed output with enhanced processing.
        
        Args:
            x: Binary latent tensor [batch_size, latent_channels, height, width]
            
        Returns:
            Reconstructed output tensor [batch_size, output_channels, height*upscale_factor, width*upscale_factor]
        """
        # Process binary features
        features = self.binary_processor(x)
        
        # Apply attention
        features = self.attention(features)
        
        # Expand channels
        features = self.expansion(features)
        
        # Apply upsampling
        for upsample_block in self.upsampling_blocks:
            features = upsample_block(features)
        
        # Final processing
        output = self.final_processor(features)
        
        return output

class HierarchicalFusionModule(nn.Module):
    """
    Module for fusing reconstructions from different levels.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            ResidualBlock(channels, channels)
        )
    
    def forward(self, higher_level: torch.Tensor, lower_level: torch.Tensor) -> torch.Tensor:
        """
        Fuse reconstructions from different levels.
        
        Args:
            higher_level: Tensor from higher (coarser) level, upsampled to match lower level
            lower_level: Tensor from lower (finer) level
            
        Returns:
            Fused tensor
        """
        # Concatenate along channel dimension
        combined = torch.cat([higher_level, lower_level], dim=1)
        
        # Apply fusion
        return self.fusion(combined)

class ContentAdaptiveFeatureFusion(nn.Module):
    """
    Enhanced feature fusion module that adapts to the content of both inputs.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels * 2, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            ResidualBlock(channels, channels)
        )
    
    def forward(self, higher_level: torch.Tensor, lower_level: torch.Tensor) -> torch.Tensor:
        """
        Fuse reconstructions with content-adaptive weighting.
        
        Args:
            higher_level: Tensor from higher (coarser) level, upsampled to match lower level
            lower_level: Tensor from lower (finer) level
            
        Returns:
            Fused tensor
        """
        # Concatenate along channel dimension
        combined = torch.cat([higher_level, lower_level], dim=1)
        
        # Calculate content-dependent attention weights
        weights = self.channel_attention(combined)
        
        # Apply attention weights
        weighted_combined = combined * weights
        
        # Apply fusion
        return self.fusion(weighted_combined)

class CrossLevelAttention(nn.Module):
    """
    Cross-level attention module to improve information flow between levels.
    """
    def __init__(self, source_channels: int, target_channels: int):
        super().__init__()
        self.source_channels = source_channels
        self.target_channels = target_channels
        self.attention = AttentionBlock(source_channels)
    
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Apply attention from source to target level.
        
        Args:
            source: Tensor from source level
            target: Tensor from target level
            
        Returns:
            Attention-enhanced tensor
        """
        # Apply attention from source to target
        attention_output = self.attention(source)
        
        # Resize attention output to match target shape
        if attention_output.shape[2:] != target.shape[2:]:
            attention_output = F.interpolate(
                attention_output,
                size=target.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Apply attention to target
        out = attention_output * target
        
        return out

class HierarchicalDecoder(nn.Module):
    """
    Enhanced hierarchical decoder with adaptive fusion and cross-level attention mechanisms.
    Supports dynamic/adaptive hierarchical levels and improved information flow.
    """
    def __init__(
        self,
        output_channels: int = 3,
        hidden_channels: int = 128,
        latent_dims: List[int] = [32, 16, 8],
        upscale_factors: List[int] = [8, 4, 2],
        max_levels: int = 4,
        use_attention: bool = True,
        use_fusion: bool = True,
        use_adaptive_fusion: bool = True,
        use_cross_level_attention: bool = True,
        final_refinement: bool = True
    ):
        """
        Initialize the hierarchical decoder.
        
        Args:
            output_channels: Number of output channels
            hidden_channels: Number of hidden channels
            latent_dims: List of latent dimensions for each level
            upscale_factors: List of spatial upscaling factors for each level
            max_levels: Maximum number of hierarchical levels to support
            use_attention: Whether to use attention blocks
            use_fusion: Whether to use fusion between levels
            use_adaptive_fusion: Whether to use content-adaptive fusion
            use_cross_level_attention: Whether to use cross-level attention mechanisms
            final_refinement: Whether to apply a final refinement network
        """
        super().__init__()
        assert len(latent_dims) == len(upscale_factors), "Must provide same number of latent dims and upscale factors"
        
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.base_latent_dims = latent_dims
        self.base_upscale_factors = upscale_factors
        self.num_base_levels = len(latent_dims)
        self.max_levels = max_levels
        self.use_fusion = use_fusion
        self.use_adaptive_fusion = use_adaptive_fusion
        self.use_cross_level_attention = use_cross_level_attention
        
        # Prepare for potential additional adaptive levels
        extended_latent_dims = latent_dims.copy()
        extended_upscale_factors = upscale_factors.copy()
        
        # Add potential additional levels
        if self.num_base_levels < max_levels:
            # For additional levels, use half the channels of previous level
            # and double the upscale factor
            for i in range(self.num_base_levels, max_levels):
                new_latent_dim = max(4, extended_latent_dims[-1] // 2)
                new_upscale = extended_upscale_factors[-1] * 2
                extended_latent_dims.append(new_latent_dim)
                extended_upscale_factors.append(new_upscale)
        
        # Store extended dimensions
        self.extended_latent_dims = extended_latent_dims
        self.extended_upscale_factors = extended_upscale_factors
        
        # Enhanced decoders for each level (from coarsest to finest)
        self.levels = nn.ModuleList()
        for i, (latent_dim, upscale) in enumerate(zip(extended_latent_dims, extended_upscale_factors)):
            # Use larger channel multiplier for coarser levels that need to generate more detail
            channel_multiplier = 8 if i == 0 else 4
            self.levels.append(
                HierarchicalLevelDecoder(
                    latent_channels=latent_dim,
                    hidden_channels=hidden_channels,
                    output_channels=output_channels,
                    upscale_factor=upscale,
                    use_attention=use_attention,
                    channel_multiplier=channel_multiplier
                )
            )
        
        # Create fusion modules if fusion is enabled
        if use_fusion:
            if use_adaptive_fusion:
                # Content-adaptive fusion modules
                self.fusion_modules = nn.ModuleList([
                    ContentAdaptiveFeatureFusion(output_channels)
                    for _ in range(max_levels - 1)  # One fusion module between each adjacent level pair
                ])
            else:
                # Standard fusion modules
                self.fusion_modules = nn.ModuleList([
                    HierarchicalFusionModule(output_channels)
                    for _ in range(max_levels - 1)
                ])
        
        # Create cross-level attention modules if enabled
        if use_cross_level_attention:
            self.cross_level_attention = nn.ModuleList([
                nn.ModuleList([
                    # Attention from source level to target level (if source < target)
                    CrossLevelAttention(
                        source_channels=output_channels,
                        target_channels=output_channels
                    ) if source_idx < target_idx else None
                    for source_idx in range(max_levels)
                ])
                for target_idx in range(max_levels)
            ])
        
        # Optional final refinement for better perceptual quality
        self.final_refinement = None
        if final_refinement:
            self.final_refinement = nn.Sequential(
                nn.Conv2d(output_channels, hidden_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                ResidualBlock(hidden_channels, hidden_channels),
                nn.Conv2d(hidden_channels, output_channels, kernel_size=3, padding=1),
                nn.Tanh()
            )
        
        # Create level compatibility conversion modules for handling arbitrary level combinations
    
    def apply_cross_level_attention(self, outputs: List[torch.Tensor], level_indices: List[int]) -> List[torch.Tensor]:
        """
        Apply cross-level attention to improve information flow between levels.
        
        Args:
            outputs: List of decoder outputs for each level
            level_indices: Indices of active levels
            
        Returns:
            List of refined outputs with cross-level attention applied
        """
        if not self.use_cross_level_attention or len(outputs) <= 1:
            return outputs
        
        refined_outputs = []
        
        for i, (output, level_idx) in enumerate(zip(outputs, level_indices)):
            # Start with original output
            refined = output
            
            # Apply attention from all previous outputs to current output
            for j in range(i):
                source_level_idx = level_indices[j]
                target_level_idx = level_idx
                
                # Skip if we don't have attention for this pair
                if source_level_idx >= target_level_idx or self.cross_level_attention[target_level_idx][source_level_idx] is None:
                    continue
                
                # Get source output and resize to target shape if needed
                source_output = outputs[j]
                if source_output.shape[2:] != output.shape[2:]:
                    source_output = F.interpolate(
                        source_output,
                        size=output.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Apply cross-level attention
                attention_output = self.cross_level_attention[target_level_idx][source_level_idx](
                    source=source_output,
                    target=refined
                )
                refined = attention_output
            
            refined_outputs.append(refined)
        
        return refined_outputs
    
    def forward(self, latents: List[torch.Tensor], level_indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        Decode binary latent representation into reconstructed image.
        
        Args:
            latents: List of binary latent tensors from the encoder
            level_indices: Optional indices indicating which latent levels are present
                          If None, assumes consecutive levels starting from 0
            
        Returns:
            Reconstructed image tensor
        """
        # If level indices not provided, assume consecutive levels from 0
        if level_indices is None:
            level_indices = list(range(len(latents)))
        
        # Handle case where encoder used adaptive hierarchy and returned a different
        # set or number of levels than we expected in the base configuration
        outputs = []
        
        # Process each level through its decoder
        for i, (latent, level_idx) in enumerate(zip(latents, level_indices)):
            # Check if we have a decoder for this level
            if level_idx >= len(self.levels):
                continue
                
            # Decode this level
            level_output = self.levels[level_idx](latent)
            outputs.append(level_output)
        
        # If no valid outputs (should never happen), return zeros
        if len(outputs) == 0:
            batch_size = latents[0].shape[0]
            return torch.zeros(batch_size, self.output_channels, latents[0].shape[2] * self.base_upscale_factors[0], 
                              latents[0].shape[3] * self.base_upscale_factors[0], device=latents[0].device)
        
        # Apply cross-level attention if enabled
        if self.use_cross_level_attention:
            outputs = self.apply_cross_level_attention(outputs, level_indices[:len(outputs)])
        
        # Fuse outputs if using fusion and we have multiple levels
        if self.use_fusion and len(outputs) > 1:
            # Process from finest to coarsest for correct information flow
            # (finest level details should influence coarser level structures)
            current_output = outputs[-1]  # Start with finest level
            
            for i in range(len(outputs) - 2, -1, -1):
                # Get current pair of outputs to fuse
                coarser_output = outputs[i]
                
                # Ensure shapes match for fusion
                if current_output.shape[2:] != coarser_output.shape[2:]:
                    # Resize to match the larger (coarser) resolution
                    current_output = F.interpolate(
                        current_output,
                        size=coarser_output.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Get fusion module for this pair
                fusion_idx = min(level_indices[i], len(self.fusion_modules) - 1)
                fusion_module = self.fusion_modules[fusion_idx]
                
                # Apply fusion
                current_output = fusion_module(coarser_output, current_output)
            
            # Use the final fused output
            final_output = current_output
        else:
            # If not using fusion, just use the coarsest level output
            final_output = outputs[0]
        
        # Apply final refinement if enabled
        if self.final_refinement is not None:
            final_output = self.final_refinement(final_output)
        
        return final_output 