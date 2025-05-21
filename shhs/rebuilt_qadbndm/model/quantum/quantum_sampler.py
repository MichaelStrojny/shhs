"""
Quantum Sampler

This module implements quantum sampling capabilities for the QADBNDM model,
using D-Wave Neal to provide simulated quantum annealing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Tuple, Dict, Optional, Union, Any

try:
    import neal
    import dimod
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    print("Warning: D-Wave Neal not available. Using classical sampling only.")

class QuantumSampler:
    """
    Quantum sampler for enhanced MCMC sampling using simulated quantum annealing.
    
    This class provides quantum sampling capabilities for binary latent diffusion,
    improving the mixing time of MCMC chains and enabling better exploration
    of the binary latent space.
    """
    def __init__(
        self,
        latent_size: int,
        max_batch_size: int = 5,
        num_anneals: int = 100,
        annealing_time: float = 20.0,
        sweep_factor: float = 10.0,
        multi_anneal: bool = False,
        anneals_per_run: int = 10,
        top_k_percent: int = 20,
        target_batch_size: int = 4,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the quantum sampler.
        
        Args:
            latent_size: Maximum size of the latent space
            max_batch_size: Maximum number of DBNs to process in a single quantum annealing run
            num_anneals: Number of annealing runs per sample
            annealing_time: Annealing time in microseconds
            sweep_factor: Factor for determining number of sweeps in simulated annealing
            multi_anneal: Whether to use multi-anneal approach with selection of best results
            anneals_per_run: Number of anneals to perform per quantum annealing run
            top_k_percent: Percentage of top results to keep from multiple anneals
            target_batch_size: Target batch size for generated samples
            device: Device to use for tensor operations
        """
        self.latent_size = latent_size
        self.max_batch_size = max_batch_size
        self.num_anneals = num_anneals
        self.annealing_time = annealing_time
        self.sweep_factor = sweep_factor
        self.multi_anneal = multi_anneal
        self.anneals_per_run = anneals_per_run
        self.top_k_percent = top_k_percent
        self.target_batch_size = target_batch_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Level-specific priorities for allocating quantum resources
        self.level_priorities = None
        
        # Initialize the quantum sampler if available
        if QUANTUM_AVAILABLE:
            self.sampler = neal.SimulatedAnnealingSampler()
        else:
            self.sampler = None
    
    def configure(
        self,
        num_anneals: Optional[int] = None,
        anneal_time: Optional[float] = None,
        level_priorities: Optional[List[float]] = None
    ):
        """
        Configure the quantum sampler with new parameters.
        
        Args:
            num_anneals: Number of annealing runs per sample
            anneal_time: Annealing time in microseconds
            level_priorities: Priority weights for each hierarchical level
        """
        if num_anneals is not None:
            self.num_anneals = num_anneals
        if anneal_time is not None:
            self.annealing_time = anneal_time
        if level_priorities is not None:
            self.level_priorities = level_priorities
    
    def _construct_qubo_from_dbn(self, dbn: nn.Module, v: torch.Tensor) -> Tuple[dict, np.ndarray, Tuple[int, int]]:
        """
        Construct a QUBO representation of the DBN energy function.
        
        Args:
            dbn: Deep Belief Network module
            v: Visible state tensor [batch_size, visible_dim]
            
        Returns:
            Tuple of (QUBO dictionary, original visible input as ndarray, shape of visible)
        """
        # Get visible and hidden dimensions
        # Handle both DeepBeliefNetwork and CrossLevelDBN
        if hasattr(dbn, 'target_visible_dim'):
            visible_dim = dbn.target_visible_dim
            hidden_dim = dbn.target_hidden_dim
        else:
            visible_dim = dbn.visible_dim
            hidden_dim = dbn.hidden_dim
        
        # Get batch size from input
        batch_size = v.shape[0]
        if batch_size > 1:
            # We only support one sample at a time for quantum annealing
            # Take the first sample in the batch
            v = v[0].view(-1)
        else:
            v = v.view(-1)
        
        # Get weights and biases from the DBN
        weights = dbn.weight.detach().cpu().numpy()
        visible_biases = dbn.visible_bias.detach().cpu().numpy()
        hidden_biases = dbn.hidden_bias.detach().cpu().numpy()
        
        # Convert visible input to numpy
        v_np = v.detach().cpu().numpy().flatten()
        
        # Create QUBO (Quadratic Unconstrained Binary Optimization)
        Q = {}
        
        # Add bias terms for hidden units (linear terms)
        for i in range(hidden_dim):
            # In QUBO, diagonal terms represent linear biases
            Q[(i, i)] = -hidden_biases[i]
        
        # Add interaction terms between hidden-hidden (quadratic terms)
        for i in range(hidden_dim):
            for j in range(i+1, hidden_dim):
                # Calculate effective interaction between hidden units via visible units
                interaction = 0.0
                for k in range(visible_dim):
                    # This models the effective interaction through the visible units
                    interaction += weights[k, i] * weights[k, j]
                
                # Skip if interaction is negligible
                if abs(interaction) > 1e-6:
                    Q[(i, j)] = interaction
        
        # Add interactions between visible and hidden units
        for i in range(visible_dim):
            for j in range(hidden_dim):
                # Effect of visible unit i on hidden unit j's bias
                # Only for visible units that are "on" (1)
                if v_np[i] > 0.5:
                    if (j, j) in Q:
                        Q[(j, j)] -= weights[i, j]
                    else:
                        Q[(j, j)] = -weights[i, j]
        
        return Q, v_np, (batch_size, visible_dim)
    
    def _construct_cl_qubo(
        self, 
        dbn: nn.Module, 
        v: torch.Tensor, 
        source_latents: List[torch.Tensor]
    ) -> Tuple[dict, np.ndarray, Tuple[int, int]]:
        """
        Construct a QUBO representation for a Cross-Level DBN with conditioning.
        
        Args:
            dbn: Cross-Level DBN module
            v: Target visible state tensor [batch_size, visible_dim]
            source_latents: List of source latent states for conditioning
            
        Returns:
            Tuple of (QUBO dictionary, original visible input as ndarray, shape of visible)
        """
        # Get dimensions
        target_visible_dim = dbn.target_visible_dim
        target_hidden_dim = dbn.target_hidden_dim
        
        # Get batch size from input
        batch_size = v.shape[0]
        if batch_size > 1:
            # We only support one sample at a time for quantum annealing
            # Take the first sample in the batch
            v = v[0].view(-1)
        else:
            v = v.view(-1)
        
        # Prepare source latents by flattening
        source_v = [s[0].reshape(-1).detach().cpu() for s in source_latents]
        
        # Get weights and biases from the DBN
        weights = dbn.weight.detach().cpu().numpy()
        visible_biases = dbn.visible_bias.detach().cpu().numpy()
        hidden_biases = dbn.hidden_bias.detach().cpu().numpy()
        
        # Get cross-level weights
        cross_level_weights = [w.detach().cpu().numpy() for w in dbn.cross_level_weights]
        cross_hidden_weights = [w.detach().cpu().numpy() for w in dbn.cross_hidden_weights]
        
        # Convert visible input to numpy
        v_np = v.detach().cpu().numpy().flatten()
        source_v_np = [s.numpy().flatten() for s in source_v]
        
        # Create QUBO
        Q = {}
        
        # Add bias terms for hidden units (linear terms)
        for i in range(target_hidden_dim):
            # Base hidden bias
            bias = hidden_biases[i]
            
            # Add cross-level influence to hidden bias
            for level_idx, s_v in enumerate(source_v_np):
                cross_weights = cross_hidden_weights[level_idx]
                for j, val in enumerate(s_v):
                    if val > 0.5:  # If source unit is on
                        bias += cross_weights[j, i]
            
            # Set bias in QUBO
            Q[(i, i)] = -bias
        
        # Add interaction terms between hidden-hidden (quadratic terms)
        for i in range(target_hidden_dim):
            for j in range(i+1, target_hidden_dim):
                # Calculate effective interaction between hidden units via visible units
                interaction = 0.0
                for k in range(target_visible_dim):
                    # Standard DBN interaction through the visible units
                    interaction += weights[k, i] * weights[k, j]
                
                # Skip if interaction is negligible
                if abs(interaction) > 1e-6:
                    Q[(i, j)] = interaction
        
        # Add interactions between visible and hidden units
        for i in range(target_visible_dim):
            for j in range(target_hidden_dim):
                # Effect of visible unit i on hidden unit j's bias
                # Only for visible units that are "on" (1)
                if v_np[i] > 0.5:
                    if (j, j) in Q:
                        Q[(j, j)] -= weights[i, j]
                    else:
                        Q[(j, j)] = -weights[i, j]
        
        return Q, v_np, (batch_size, target_visible_dim)
    
    def _sample_from_qubo(
        self, 
        Q: dict, 
        num_reads: int = 100
    ) -> np.ndarray:
        """
        Sample from a QUBO model using simulated quantum annealing.
        
        Args:
            Q: QUBO dictionary
            num_reads: Number of samples to generate
            
        Returns:
            Sampled binary states
        """
        if not QUANTUM_AVAILABLE:
            # Fallback to standard sampling if quantum not available
            hidden_dim = max(max(Q.keys())) + 1
            return np.random.randint(0, 2, size=(num_reads, hidden_dim))
        
        # Determine number of sweeps based on problem size and factor
        num_vars = max(max(Q.keys())) + 1 if Q else 0
        num_sweeps = int(np.sqrt(num_vars) * self.sweep_factor)
        
        # Create BQM from QUBO
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
        
        # Sample from the model
        response = self.sampler.sample(
            bqm,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            beta_range=(0.1, 1000.0)
        )
        
        # Convert samples to numpy array
        samples = np.zeros((num_reads, num_vars), dtype=np.int8)
        
        for i, sample in enumerate(response.record):
            if i >= num_reads:
                break
            samples[i] = sample[0]
        
        return samples
    
    def _reconstruct_visible(
        self, 
        h_samples: np.ndarray, 
        dbn: nn.Module, 
        original_v: np.ndarray,
        orig_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Reconstruct visible units from hidden samples.
        
        Args:
            h_samples: Hidden state samples
            dbn: Deep Belief Network module
            original_v: Original visible state input
            orig_shape: Original shape of visible input
            
        Returns:
            Reconstructed visible state tensor
        """
        # Get weights and biases from the DBN
        weights = dbn.weight.detach().cpu().numpy()
        visible_biases = dbn.visible_bias.detach().cpu().numpy()
        
        # Create hidden state tensor
        h_tensor = torch.tensor(h_samples[0], dtype=torch.float32, device=self.device).view(1, -1)
        
        # Use DBN to generate visible state
        with torch.no_grad():
            # Move hidden state to DBN's device
            h_tensor = h_tensor.to(next(dbn.parameters()).device)
            
            # Generate visible state from hidden state
            v_prob, v_samples = dbn.sample_v_given_h(h_tensor)
        
        # Reshape to original format
        v_samples = v_samples.view(orig_shape)
        
        return v_samples
    
    def _reconstruct_cl_visible(
        self, 
        h_samples: np.ndarray, 
        dbn: nn.Module, 
        original_v: np.ndarray,
        orig_shape: Tuple[int, int],
        source_latents: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Reconstruct visible units from hidden samples for a Cross-Level DBN.
        
        Args:
            h_samples: Hidden state samples
            dbn: Cross-Level DBN module
            original_v: Original visible state input
            orig_shape: Original shape of visible input
            source_latents: List of source latent states for conditioning
            
        Returns:
            Reconstructed visible state tensor
        """
        # Prepare source visible states by flattening
        source_v = [s[0].reshape(1, -1).detach() for s in source_latents]
        
        # Create hidden state tensor
        h_tensor = torch.tensor(h_samples[0], dtype=torch.float32, device=self.device).view(1, -1)
        
        # Use DBN to generate visible state
        with torch.no_grad():
            # Move hidden state to DBN's device
            h_tensor = h_tensor.to(next(dbn.parameters()).device)
            
            # Generate visible state from hidden state with conditioning
            v_prob, v_samples = dbn.sample_v_given_h(h_tensor, source_v)
        
        # Reshape to original format
        v_samples = v_samples.view(orig_shape)
        
        return v_samples
    
    def _sample_dbn_multi_anneal(
        self, 
        dbn: nn.Module, 
        v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample from a DBN using a multi-anneal approach for better results.
        
        Args:
            dbn: Deep Belief Network module
            v: Visible state tensor
            
        Returns:
            Tuple of (visible_probabilities, visible_samples, hidden_probabilities)
        """
        # Determine how many anneals to perform
        num_anneals = self.anneals_per_run
        
        # Flatten input if needed
        if len(v.shape) > 2:
            v_flat = v.reshape(v.shape[0], -1)
        else:
            v_flat = v
        
        # Construct QUBO
        Q, original_v, orig_shape = self._construct_qubo_from_dbn(dbn, v_flat)
        
        # Sample multiple times
        h_samples_multi = self._sample_from_qubo(Q, num_reads=num_anneals)
        
        # Calculate energy for each sample
        energies = []
        for i in range(num_anneals):
            h_sample = h_samples_multi[i]
            energy = 0.0
            
            # Calculate energy from QUBO
            for (i, j), weight in Q.items():
                if i == j:
                    energy += weight * h_sample[i]
                else:
                    energy += weight * h_sample[i] * h_sample[j]
            
            energies.append(energy)
        
        # Select top samples based on energy (lower is better)
        top_k = max(1, int(num_anneals * self.top_k_percent / 100))
        top_indices = np.argsort(energies)[:top_k]
        
        # Get best sample
        best_h_sample = h_samples_multi[top_indices[0]]
        
        # Create hidden state tensor
        h_tensor = torch.tensor(best_h_sample, dtype=torch.float32, device=self.device).view(1, -1)
        
        # Use DBN to generate visible state
        with torch.no_grad():
            # Move hidden state to DBN's device
            h_tensor = h_tensor.to(next(dbn.parameters()).device)
            
            # Get hidden probabilities
            h_prob = torch.sigmoid(F.linear(v_flat, dbn.weight.t(), dbn.hidden_bias))
            
            # Generate visible state from hidden state
            v_prob, v_samples = dbn.sample_v_given_h(h_tensor)
        
        return v_prob, v_samples, h_prob
    
    def sample_dbn(
        self, 
        dbn: nn.Module, 
        v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample from a DBN using quantum annealing.
        
        Args:
            dbn: Deep Belief Network module
            v: Visible state tensor
            
        Returns:
            Tuple of (visible_probabilities, visible_samples, hidden_probabilities)
        """
        # Check if we should use multi-anneal approach
        if self.multi_anneal:
            return self._sample_dbn_multi_anneal(dbn, v)
        
        # Flatten input if needed
        if len(v.shape) > 2:
            v_flat = v.reshape(v.shape[0], -1)
        else:
            v_flat = v
        
        # Construct QUBO
        Q, original_v, orig_shape = self._construct_qubo_from_dbn(dbn, v_flat)
        
        # Sample from QUBO
        h_samples = self._sample_from_qubo(Q, num_reads=self.num_anneals)
        
        # Reconstruct visible state
        v_samples = self._reconstruct_visible(h_samples, dbn, original_v, orig_shape)
        
        # Get probabilities
        with torch.no_grad():
            # Calculate hidden probabilities
            h_prob = torch.sigmoid(F.linear(v_flat, dbn.weight.t(), dbn.hidden_bias))
            
            # Calculate visible probabilities
            v_prob = torch.sigmoid(F.linear(h_prob, dbn.weight, dbn.visible_bias))
        
        return v_prob, v_samples, h_prob
    
    def sample_cl_dbn(
        self, 
        dbn: nn.Module, 
        v: torch.Tensor, 
        source_latents: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample from a Cross-Level DBN using quantum annealing.
        
        Args:
            dbn: Cross-Level DBN module
            v: Target visible state tensor
            source_latents: List of source latent states for conditioning
            
        Returns:
            Tuple of (visible_probabilities, visible_samples, hidden_probabilities)
        """
        # Flatten input if needed
        if len(v.shape) > 2:
            v_flat = v.reshape(v.shape[0], -1)
        else:
            v_flat = v
        
        # Prepare source visible states by flattening if needed
        source_v = [s.reshape(s.shape[0], -1) for s in source_latents]
        
        # Construct QUBO with cross-level conditioning
        Q, original_v, orig_shape = self._construct_cl_qubo(dbn, v_flat, source_latents)
        
        # Sample from QUBO
        h_samples = self._sample_from_qubo(Q, num_reads=self.num_anneals)
        
        # Reconstruct visible state with conditioning
        v_samples = self._reconstruct_cl_visible(h_samples, dbn, original_v, orig_shape, source_latents)
        
        # Get probabilities
        with torch.no_grad():
            # Calculate hidden probabilities with conditioning
            h_prob, _ = dbn.sample_h_given_v(v_flat, source_v)
            
            # Calculate visible probabilities with conditioning
            v_prob, _ = dbn.sample_v_given_h(h_prob, source_v)
        
        return v_prob, v_samples, h_prob
    
    def sample_dbns_batch(
        self, 
        dbns: List[nn.Module], 
        inputs: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Sample from multiple DBNs in a single quantum annealing run.
        
        Args:
            dbns: List of DBN modules
            inputs: List of input tensors
            
        Returns:
            List of visible sample tensors
        """
        num_dbns = len(dbns)
        
        # Ensure we don't exceed max batch size
        if num_dbns > self.max_batch_size:
            # Split into multiple batches
            results = []
            for i in range(0, num_dbns, self.max_batch_size):
                batch_dbns = dbns[i:i+self.max_batch_size]
                batch_inputs = inputs[i:i+self.max_batch_size]
                
                # Process this batch
                batch_results = self.sample_dbns_batch(batch_dbns, batch_inputs)
                results.extend(batch_results)
            
            return results
        
        # Process DBNs in a batch
        samples_list = []
        
        for i, (dbn, input_tensor) in enumerate(zip(dbns, inputs)):
            # Sample from this DBN
            if hasattr(dbn, 'target_visible_dim'):
                # This is a CrossLevelDBN, we need source latents
                # Since we don't have them here, fall back to standard sampling
                _, samples, _ = dbn(input_tensor, k=1)
            else:
                # Standard DBN
                _, samples, _ = self.sample_dbn(dbn, input_tensor)
            
            samples_list.append(samples)
        
        return samples_list 