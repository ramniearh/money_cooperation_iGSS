"""
Example usage of the IGSS Framework.

This script demonstrates how to use the unified framework to run
all experimental modes with exact functional equivalence to the
original 11 separate scripts.
"""

import numpy as np
from igss_framework import (
    run_experiment,
    list_experiments,
    ExperimentRegistry,
    ExperimentConfig,
    ModelConfig,
    EvoConfig,
    PrimitiveConfig,
)


def example_1_run_single_experiment():
    """Example 1: Run a single experiment."""
    print("=" * 60)
    print("Example 1: Running DR_mode1")
    print("=" * 60)
    
    result = run_experiment("DR_mode1", verbose=True, show_plots=False)
    
    print(f"\nResults:")
    print(f"  Best fitness: {result.history.max_igss[-1]:.2f}")
    print(f"  Control baseline: {result.control_fitness:.2f}")
    print(f"  Generations: {len(result.history.max_igss)}")


def example_2_list_all_experiments():
    """Example 2: List all available experiments."""
    print("\n" + "=" * 60)
    print("Example 2: Available Experiments")
    print("=" * 60)
    
    experiments = list_experiments()
    for name in experiments:
        print(f"  - {name}")


def example_3_run_multiple_experiments():
    """Example 3: Run multiple experiments and compare."""
    print("\n" + "=" * 60)
    print("Example 3: Comparing DR Modes")
    print("=" * 60)
    
    results = {}
    for mode in ["DR_mode1", "DR_mode2", "DR_mode3"]:
        print(f"\nRunning {mode}...")
        result = run_experiment(mode, verbose=False, show_plots=False)
        results[mode] = result
        print(f"  {mode}: {result.history.max_igss[-1]:.2f}")
    
    print("\nComparison:")
    for mode, result in results.items():
        final_fit = result.history.max_igss[-1]
        control = result.control_fitness
        improvement = ((final_fit - control) / control * 100) if control > 0 else 0
        print(f"  {mode}: {final_fit:.2f} (vs control {control:.2f}, {improvement:+.1f}%)")


def example_4_custom_configuration():
    """Example 4: Create and run a custom experiment configuration."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Configuration")
    print("=" * 60)
    
    # Create custom configuration
    custom_config = ExperimentConfig(
        name="Custom_DR_Experiment",
        mode="mode1",
        mechanism="DR",
        model=ModelConfig(
            benefit_to_cost_ratio=3,  # Lower benefit/cost
            num_igss=20,              # More iGSS agents
            num_rounds=50,            # Shorter rounds
        ),
        evolution=EvoConfig(
            pop_size=30,              # Smaller population
            max_gens=50,              # Fewer generations
        ),
        action_primitive=PrimitiveConfig.dr_action(),
        seed=123,                     # Different seed
    )
    
    # Register and run
    ExperimentRegistry.register(custom_config.name, custom_config)
    result = run_experiment(custom_config.name, verbose=True, show_plots=False)
    
    print(f"\nCustom experiment completed!")
    print(f"  Final fitness: {result.history.max_igss[-1]:.2f}")


def example_5_access_detailed_results():
    """Example 5: Access detailed results from an experiment."""
    print("\n" + "=" * 60)
    print("Example 5: Detailed Results Access")
    print("=" * 60)
    
    result = run_experiment("IR_mode1", verbose=False, show_plots=False)
    
    # Access history
    history = result.history
    print(f"\nEvolution History:")
    print(f"  Max fitness per generation: {len(history.max_igss)} points")
    print(f"  Average fitness per generation: {len(history.avg_igss)} points")
    print(f"  Fossil record entries: {len(history.fossil_record)}")
    
    # Access configuration
    config = result.config
    print(f"\nConfiguration:")
    print(f"  Mechanism: {config.mechanism}")
    print(f"  Mode: {config.mode}")
    print(f"  Seed: {config.seed}")
    print(f"  Population size: {config.evolution.pop_size}")
    
    # Access best individual
    best = result.best_individual
    print(f"\nBest Individual:")
    print(f"  Raw: {str(best)[:80]}...")


def example_6_batch_processing():
    """Example 6: Batch processing without visualization."""
    print("\n" + "=" * 60)
    print("Example 6: Batch Processing")
    print("=" * 60)
    
    # Run multiple seeds
    seeds = [42, 43, 44]
    results = []
    
    for seed in seeds:
        # Create config with different seed
        config = ExperimentConfig(
            name=f"Batch_seed_{seed}",
            mode="mode1",
            mechanism="DR",
            model=ModelConfig(),
            evolution=EvoConfig(max_gens=20),  # Short for demo
            action_primitive=PrimitiveConfig.dr_action(),
            seed=seed,
        )
        
        ExperimentRegistry.register(config.name, config)
        result = run_experiment(config.name, verbose=False, show_plots=False)
        results.append(result.history.max_igss[-1])
        print(f"  Seed {seed}: {result.history.max_igss[-1]:.2f}")
    
    print(f"\nStatistics across {len(seeds)} runs:")
    print(f"  Mean: {np.mean(results):.2f}")
    print(f"  Std: {np.std(results):.2f}")
    print(f"  Min: {np.min(results):.2f}")
    print(f"  Max: {np.max(results):.2f}")


def example_7_reproduce_original_paper():
    """Example 7: Reproduce exact results from original paper."""
    print("\n" + "=" * 60)
    print("Example 7: Reproducing Original Paper Results")
    print("=" * 60)
    
    # The framework maintains exact functional equivalence
    # to the original 11 scripts through:
    # - Same random seeds
    # - Same configuration parameters
    # - Same GP operators
    # - Same simulation logic
    
    print("\nRunning all original experiments...")
    
    original_experiments = [
        "DR_mode1", "DR_mode2", "DR_mode3",
        "IR_mode1", "IR_mode2_donor", "IR_mode2_nodonor", "IR_mode3",
        "jointDRIR_mode1", "jointDRIR_mode2_AND_OR_twotrees", "jointDRIR_mode3_threetrees",
        "MONEY_mode1_integer_exclusive_defenced",
    ]
    
    for exp_name in original_experiments:
        result = run_experiment(exp_name, verbose=False, show_plots=False)
        print(f"  {exp_name}: {result.history.max_igss[-1]:.2f}")
    
    print("\nNote: Results should match original scripts exactly due to:")
    print("  - Identical random seeds")
    print("  - Identical configuration parameters")
    print("  - Identical GP operators and probabilities")
    print("  - Identical simulation resolution logic")


if __name__ == "__main__":
    # Run all examples
    print("\n" + "=" * 60)
    print("IGSS Framework - Example Usage")
    print("=" * 60)
    
    example_1_run_single_experiment()
    example_2_list_all_experiments()
    example_3_run_multiple_experiments()
    example_4_custom_configuration()
    example_5_access_detailed_results()
    example_6_batch_processing()
    example_7_reproduce_original_paper()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
