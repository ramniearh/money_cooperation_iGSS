"""
Main entry point for IGSS framework.

Provides command-line interface for running experiments.
"""

import sys
import argparse
from typing import Optional

from .experiments import ExperimentRegistry, list_experiments, run_experiment


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="IGSS Framework: Evolutionary experiments on cooperation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available experiments
  python -m igss_framework --list
  
  # Run a specific experiment
  python -m igss_framework DR_mode1
  
  # Run without visualization (for batch processing)
  python -m igss_framework IR_mode3 --no-plots
  
  # Run quietly
  python -m igss_framework jointDRIR_mode3_threetrees --quiet
"""
    )
    
    parser.add_argument(
        "experiment",
        nargs="?",
        help="Name of experiment to run"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available experiments"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable visualization (for batch processing)"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="%(prog)s 1.0.0"
    )
    
    return parser


def main(args: Optional[list] = None) -> int:
    """
    Main entry point.
    
    Args:
        args: Command-line arguments (defaults to sys.argv)
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = create_parser()
    parsed = parser.parse_args(args)
    
    # List experiments
    if parsed.list:
        print("Available experiments:")
        for name in list_experiments():
            print(f"  - {name}")
        return 0
    
    # Require experiment name
    if not parsed.experiment:
        parser.print_help()
        print("\nError: Experiment name required (use --list to see available experiments)")
        return 1
    
    # Run experiment
    try:
        result = run_experiment(
            name=parsed.experiment,
            verbose=not parsed.quiet,
            show_plots=not parsed.no_plots
        )
        
        if not parsed.quiet:
            print(f"\nExperiment completed successfully!")
            print(f"Final iGSS fitness: {result.history.max_igss[-1]:.2f}")
            print(f"Control baseline: {result.control_fitness:.2f}")
        
        return 0
        
    except KeyError as e:
        print(f"Error: {e}")
        print(f"\nRun with --list to see available experiments")
        return 1
    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
