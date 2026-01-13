"""
Script to batch-generate UMAPs for all experiments in a directory tree.

Usage:
    python all_umaps.py --base_dir experiments/my_runs/
    python all_umaps.py --base_dir experiments/ --target_epoch 50
    python all_umaps.py --base_dir experiments/ --pattern "*test*"
"""

import os
import argparse
import sys
from pathlib import Path
from typing import List, Optional

from fcr.validation.view_results import plot_umaps
from fcr.validation.view_results import plot_progression

from view_umaps import output_umaps


def find_experiment_directories(base_dir: str, pattern: Optional[str] = None) -> List[str]:
    """
    Find all directories that contain model checkpoints (saves/ subdirectory).
    
    Parameters
    ----------
    base_dir : str
        Root directory to search
    pattern : str, optional
        Glob pattern to filter experiment names (e.g., "*test*", "run_*")
        
    Returns
    -------
    list of str
        List of experiment directory paths
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory does not exist: {base_dir}")
    
    experiment_dirs = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_path):
        root_path = Path(root)
        #print(root_path)

        # Check if this directory contains a 'saves' subdirectory (indicating model checkpoints)
        saves_dir = root_path / "saves"
        if saves_dir.exists() and saves_dir.is_dir():
            # Check pattern if it exists
            if pattern is None or root_path.match(pattern):
                experiment_dirs.append(str(root_path))
                print(f"Found experiment: {root_path}")
    
    return sorted(experiment_dirs)


def process_experiments(
    base_dir: str,
    sample: bool = False,
    target_epoch: Optional[int] = None,
    plot_prog: bool = False,
    pattern: Optional[str] = None,
    skip_errors: bool = True,
    verbose: bool = True,
    drug: Optional[str] = None,
    all_drugs: Optional[bool] = False,
    cell: Optional[str] = None,
) -> dict:
    """
    Process all experiments in a directory tree and generate UMAPs.
    
    Parameters
    ----------
    base_dir : str
        Root directory containing experiment subdirectories
    target_epoch : int, optional
        Specific epoch to plot. If None, uses latest checkpoint
    pattern : str, optional
        Glob pattern to filter experiment names
    skip_errors : bool, default=True
        If True, continue processing even if some experiments fail
    verbose : bool, default=True
        Print detailed progress information
        
    Returns
    -------
    dict
        Dictionary with 'success' and 'failed' lists of experiment paths
    """
    
    print(f"\n{'='*70}")
    print(f"Searching for experiments in: {base_dir}")
    print(f"{'='*70}\n")
    
    # Find all experiment directories
    experiment_dirs = find_experiment_directories(base_dir, pattern=pattern)
    
    if not experiment_dirs:
        print(f"No experiment directories found in {base_dir}")
        print("(Looking for directories with 'saves/' subdirectory containing .pt files)")
        return {'success': [], 'failed': []}
    
    print(f"\nFound {len(experiment_dirs)} experiment(s) to process")
    print(f"Target epoch: {'Latest' if target_epoch is None else target_epoch}\n")
    
    results = {
        'success': [],
        'failed': []
    }
    
    # Process each experiment
    for i, exp_dir in enumerate(experiment_dirs, 1):
        print(f"\n[{i}/{len(experiment_dirs)}] Processing: {exp_dir}")
        print("-" * 70)
        
        try:
            # Generate UMAPs for this experiment
            output_umaps(
                model_dir=exp_dir,
                target_epoch=target_epoch,
                drug=drug,
                all_drugs=all_drugs,
                cell=cell,
                plot_prog=False,
                sample=sample
            )

            results['success'].append(exp_dir)
            
            # Check if UMAPs were created
            umaps_dir = os.path.join(exp_dir, "umaps")
            if os.path.exists(umaps_dir):
                umap_files = [f for f in os.listdir(umaps_dir) if f.endswith('.png')]
                print(f"✓ Success! Generated {len(umap_files)} UMAP plot(s) in {umaps_dir}")
            else:
                print(f"✓ Completed (no umaps directory found)")

        
        except Exception as e:
            print(f"✗ Failed with error: {e}")
            results['failed'].append(exp_dir)
            
            if not skip_errors:
                raise
            else:
                print("  Continuing to next experiment...")
        
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total experiments: {len(experiment_dirs)}")
    print(f"Successful: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")
    
    if results['failed']:
        print("\nFailed experiments:")
        for exp_dir in results['failed']:
            print(f"  - {exp_dir}")
    
    print(f"{'='*70}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate UMAPs for all experiments in a directory tree",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all experiments in a directory
  python all_umaps.py --base_dir experiments/my_runs/
  
  # Process specific epoch for all experiments
  python all_umaps.py --base_dir experiments/ --target_epoch 100
  
  # Filter experiments by pattern
  python all_umaps.py --base_dir experiments/ --pattern "*test*"
  
  # Stop on first error (don't skip failures)
  python all_umaps.py --base_dir experiments/ --no-skip-errors
        """
    )
    
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Root directory containing experiment subdirectories"
    )
    
    parser.add_argument(
        "--target_epoch",
        type=int,
        default=None,
        help="Specific epoch to plot (default: latest checkpoint)"
    )

    parser.add_argument(
        "--plot_prog",
        action="store_true",
        help="If specified, plots progress of UMAP along the epochs"
    )

    parser.add_argument(
        "--sample",
        action="store_true",
        help="If specified, samples from latent distributions instead of using their means"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Glob pattern to filter experiment directories (e.g., '*test*', 'run_*')"
    )
    
    parser.add_argument(
        "--no-skip-errors",
        action="store_true",
        help="Stop processing on first error instead of continuing"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbosity"
    )

    parser.add_argument(
        "--drug",
        type=str,
        default=None,
        help="If specified, filter to only this drug"
    )

    parser.add_argument(
        "--all_drugs",
        action="store_true",
        help="If specified, filters to all drugs and generates UMAPs for each"
    )

    parser.add_argument(
        "--cell",
        type=str,
        default=None,
        help="If specified, filter to only this cell line"
    )
    
    args = parser.parse_args()
    
    # Process all experiments
    results = process_experiments(
        base_dir=args.base_dir,
        target_epoch=args.target_epoch,
        pattern=args.pattern,
        skip_errors=not args.no_skip_errors,
        verbose=not args.quiet,
        drug=args.drug,
        all_drugs=args.all_drugs,
        cell=args.cell,
        sample=args.sample,
        plot_prog=args.plot_prog
    )
    
    # Exit with error code if any experiments failed
    if results['failed']:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
