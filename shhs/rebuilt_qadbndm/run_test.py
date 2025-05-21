#!/usr/bin/env python
"""
QADBNDM Test Run Script

This script runs a complete test of the QADBNDM training and evaluation pipeline
using a small subset of CIFAR-10 data with multiple quantum configurations.
"""

import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('qadbndm_test_run')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run QADBNDM test suite")
    parser.add_argument("--output-dir", type=str, default="test_outputs",
                      help="Directory to save outputs")
    parser.add_argument("--qubit-counts", nargs="+", type=int, default=[2048, 5760],
                      help="Qubit counts to test with")
    parser.add_argument("--multi-anneal", action="store_true",
                      help="Use multi-anneal quantum sampling")
    parser.add_argument("--dataset", type=str, default="cifar10", 
                      choices=["cifar10", "mnist", "fashion_mnist"],
                      help="Dataset to test on")
    parser.add_argument("--epochs", type=int, default=1,
                      help="Number of epochs for training test")
    parser.add_argument("--train-only", action="store_true",
                      help="Run only the training test")
    parser.add_argument("--test-only", action="store_true",
                      help="Run only the testing with saved checkpoint")
    parser.add_argument("--checkpoint", type=str,
                      help="Path to checkpoint for testing (required if --test-only)")
    return parser.parse_args()

def run_training_test(qubit_count, dataset, output_dir, multi_anneal=False, epochs=1):
    """
    Run a training test with the specified configuration.
    
    Args:
        qubit_count: Number of qubits for configuration
        dataset: Dataset to train on
        output_dir: Output directory
        multi_anneal: Whether to use multi-anneal sampling
        epochs: Number of epochs to train for
        
    Returns:
        Path to saved checkpoint
    """
    logger.info(f"Running training test with {qubit_count} qubits on {dataset}")
    
    # Build command
    cmd = [
        "python", "-m", "rebuilt_qadbndm.train",
        "--qubit-count", str(qubit_count),
        "--dataset", dataset,
        "--epochs", str(epochs),
        "--save-dir", output_dir,
        "--batch-size", "32",
        "--sample-interval", "1",
        "--checkpoint-interval", "1",
        "--test-mode"
    ]
    
    if multi_anneal:
        cmd.append("--multi-anneal")
    
    # Log command
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run command
    process = subprocess.run(cmd, check=True)
    
    # Find latest checkpoint
    checkpoints = list(Path(output_dir).glob(f"{dataset}_{qubit_count}qubits_*/**/best_model.pt"))
    if not checkpoints:
        checkpoints = list(Path(output_dir).glob(f"{dataset}_{qubit_count}qubits_*/**/checkpoint_*.pt"))
    
    if not checkpoints:
        logger.warning("No checkpoint found after training")
        return None
    
    # Use the most recent checkpoint
    checkpoint_path = str(sorted(checkpoints, key=os.path.getmtime)[-1])
    logger.info(f"Training completed, checkpoint saved at: {checkpoint_path}")
    
    return checkpoint_path

def run_testing(checkpoint_path, qubit_counts, dataset, output_dir, multi_anneal=False):
    """
    Run testing with the specified configuration.
    
    Args:
        checkpoint_path: Path to model checkpoint
        qubit_counts: List of qubit counts to test with
        dataset: Dataset to test on
        output_dir: Output directory
        multi_anneal: Whether to use multi-anneal comparison
    """
    logger.info(f"Running testing with checkpoint: {checkpoint_path}")
    
    # Build command
    cmd = [
        "python", "-m", "rebuilt_qadbndm.test",
        "--checkpoint", checkpoint_path,
        "--dataset", dataset,
        "--output-dir", output_dir,
        "--batch-size", "4",
        "--steps", "8"
    ]
    
    # Add qubit counts
    if not multi_anneal:
        cmd.extend(["--qubit-counts"] + [str(qc) for qc in qubit_counts])
    else:
        cmd.append("--compare-multi-anneal")
        cmd.extend(["--anneals-per-run", "10", "--top-k-percent", "20.0"])
    
    # Log command
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run command
    process = subprocess.run(cmd, check=True)
    
    logger.info("Testing completed")

def main():
    """Main function to run test suite."""
    args = parse_args()
    
    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        args.output_dir, 
        f"complete_test_{args.dataset}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Running QADBNDM test suite with output to: {output_dir}")
    logger.info(f"Qubit counts: {args.qubit_counts}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Multi-anneal: {args.multi_anneal}")
    
    checkpoint_path = None
    
    # Run training if not test-only
    if not args.test_only:
        logger.info("Starting training test...")
        # Use first qubit count for training
        checkpoint_path = run_training_test(
            args.qubit_counts[0], 
            args.dataset, 
            output_dir, 
            args.multi_anneal,
            args.epochs
        )
    
    # Run testing if not train-only and we have a checkpoint
    if not args.train_only:
        # Use provided checkpoint or the one from training
        checkpoint_to_use = args.checkpoint if args.test_only else checkpoint_path
        
        if checkpoint_to_use:
            logger.info("Starting testing...")
            run_testing(
                checkpoint_to_use,
                args.qubit_counts,
                args.dataset,
                output_dir,
                args.multi_anneal
            )
        else:
            logger.error("No checkpoint available for testing")
    
    logger.info(f"Test suite completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 