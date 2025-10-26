#!/usr/bin/env python3
"""
Main execution script for Brain Connectivity-Based Cognitive Prediction.

This script runs the complete analysis pipeline:
1. Load/generate data
2. Preprocess connectivity matrices
3. Train prediction models
4. Evaluate performance
5. Perform statistical tests
6. Generate visualizations
7. Create HTML report

Usage:
    python run_pipeline.py --config config.yaml
    python run_pipeline.py --config config.yaml --quick  # Quick test with fewer iterations

Author: Based on Dhamala et al. (2021)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import logging
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger
from src.utils.io import load_config, save_results, ensure_dir
from src.data.generate_synthetic import generate_synthetic_data
from src.data.preprocess import preprocess_connectivity, create_hybrid_connectivity
from src.data.load_connectivity import matrices_to_feature_vectors
from src.models.ridge_prediction import train_test_prediction
from src.models.permutation_test import run_permutation_tests
from src.models.feature_importance import compute_feature_importance, feature_vector_to_matrix
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.compare_models import compare_all_models, fdr_correction
from src.visualization.plot_results import (
    plot_performance_violin,
    plot_performance_boxplot,
    plot_scatter_prediction,
    plot_comparison_heatmap
)
from src.visualization.plot_features import (
    plot_feature_importance_matrix,
    plot_regional_importance,
    plot_top_connections
)
from src.visualization.report_generator import generate_html_report


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Brain Connectivity-Based Cognitive Prediction Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (reduces iterations for faster execution)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def setup_logging(config: dict, verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    log_config = config.get('logging', {})
    log_level = "DEBUG" if verbose else log_config.get('level', 'INFO')
    log_file = log_config.get('file', 'outputs/logs/pipeline.log')
    
    logger = setup_logger(
        name='pipeline',
        log_file=log_file,
        level=log_level,
        console=True
    )
    
    return logger


def main():
    """Main pipeline execution."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Quick mode modifications
    if args.quick:
        print("\n" + "=" * 80)
        print("QUICK TEST MODE ENABLED")
        print("=" * 80)
        config['model']['train_test']['n_splits'] = 10
        config['statistics']['permutation']['n_permutations'] = 100
        config['data']['n_subjects'] = 100
    
    # Override output directory if specified
    if args.output_dir:
        for key in config['output']['dirs']:
            config['output']['dirs'][key] = str(Path(args.output_dir) / Path(config['output']['dirs'][key]).name)
    
    # Setup logging
    logger = setup_logging(config, args.verbose)
    
    logger.info("=" * 80)
    logger.info("BRAIN CONNECTIVITY-BASED COGNITIVE PREDICTION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Output directory: {config['output']['dirs']['results']}")
    
    # Create output directories
    for dir_path in config['output']['dirs'].values():
        ensure_dir(dir_path)
    
    # Set random seed for reproducibility
    if config['reproducibility']['set_seeds']:
        seed = config['reproducibility']['global_seed']
        np.random.seed(seed)
        logger.info(f"Random seed set to: {seed}")
    
    # =========================================================================
    # STEP 1: LOAD/GENERATE DATA
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: DATA LOADING/GENERATION")
    logger.info("=" * 80)
    
    if config['data']['synthetic']['use_synthetic']:
        logger.info("Generating synthetic data...")
        data = generate_synthetic_data(
            n_subjects=config['data']['n_subjects'],
            n_regions=config['data']['n_regions'],
            config=config,
            seed=config['reproducibility']['global_seed']
        )
    else:
        logger.error("Real data loading not yet implemented. Please use synthetic data.")
        sys.exit(1)
    
    fc_matrices = data['FC']
    sc_matrices = data['SC']
    cognitive_scores = data['cognitive_scores']
    score_names = data['score_names']
    subject_ids = data['subject_ids']
    
    logger.info(f"Loaded data for {len(subject_ids)} subjects")
    logger.info(f"FC shape: {fc_matrices.shape}")
    logger.info(f"SC shape: {sc_matrices.shape}")
    logger.info(f"Cognitive scores shape: {cognitive_scores.shape}")
    
    # =========================================================================
    # STEP 2: PREPROCESS DATA
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: DATA PREPROCESSING")
    logger.info("=" * 80)
    
    # Preprocess FC
    logger.info("Preprocessing functional connectivity...")
    fc_processed = preprocess_connectivity(fc_matrices, "FC", config)
    
    # Preprocess SC
    logger.info("Preprocessing structural connectivity...")
    sc_processed = preprocess_connectivity(sc_matrices, "SC", config)
    
    # Create hybrid connectivity
    logger.info("Creating hybrid connectivity...")
    hc_processed = create_hybrid_connectivity(
        fc_processed,
        sc_processed,
        method=config['preprocessing']['hc']['concatenation'],
        fc_weight=config['preprocessing']['hc']['fc_weight'],
        sc_weight=config['preprocessing']['hc']['sc_weight']
    )
    
    # Convert to feature vectors
    logger.info("Converting matrices to feature vectors...")
    X_fc = matrices_to_feature_vectors(fc_processed, vectorize=True)
    X_sc = matrices_to_feature_vectors(sc_processed, vectorize=True)
    X_hc = matrices_to_feature_vectors(hc_processed, vectorize=True)
    
    logger.info(f"FC features shape: {X_fc.shape}")
    logger.info(f"SC features shape: {X_sc.shape}")
    logger.info(f"HC features shape: {X_hc.shape}")
    
    # =========================================================================
    # STEP 3: TRAIN PREDICTION MODELS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: TRAINING PREDICTION MODELS")
    logger.info("=" * 80)
    
    # Prepare connectivity types
    X_dict = {'FC': X_fc, 'SC': X_sc, 'HC': X_hc}
    
    # Prepare cognitive targets
    y_dict = {}
    for i, score_name in enumerate(score_names):
        y_dict[score_name] = cognitive_scores[:, i]
    
    # Train models for all combinations
    results = {}
    
    for conn_type, X in X_dict.items():
        results[conn_type] = {}
        
        for score_name, y in y_dict.items():
            logger.info(f"\nTraining {conn_type} -> {score_name}") 
            
            model_results = train_test_prediction(
                X=X,
                y=y,
                alpha_range=config['model']['ridge']['alpha_range'],
                n_splits=config['model']['train_test']['n_splits'],
                test_size=config['model']['train_test']['test_size'],
                random_seed=config['model']['train_test']['random_seed']
            )
            
            results[conn_type][score_name] = model_results
            
            # Save results
            save_results(
                model_results,
                f"{config['output']['dirs']['results']}/{conn_type}_{score_name}_results.json"
            )
    
    logger.info("\nAll models trained successfully!")
    
    # =========================================================================
    # STEP 4: FEATURE IMPORTANCE
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: COMPUTING FEATURE IMPORTANCE")
    logger.info("=" * 80)
    
    feature_importance_results = {}
    
    if config['model']['feature_importance']['compute']:
        for conn_type, X in X_dict.items():
            feature_importance_results[conn_type] = {}
            
            for score_name, y in y_dict.items():
                logger.info(f"\nComputing importance for {conn_type} -> {score_name}")
                
                coefficients = results[conn_type][score_name]['coefficients']
                
                importance = compute_feature_importance(
                    X=X,
                    y=y,
                    coefficients_array=coefficients,
                    method=config['model']['feature_importance']['method']
                )
                
                feature_importance_results[conn_type][score_name] = importance
                
                # Save feature importance
                save_results(
                    importance,
                    f"{config['output']['dirs']['results']}/{conn_type}_{score_name}_importance.pkl",
                    format='pkl'
                )
    
    # =========================================================================
    # STEP 5: STATISTICAL TESTING
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: STATISTICAL TESTING")
    logger.info("=" * 80)
    
    # Model comparisons
    logger.info("\nComparing models...")
    comparison_results = compare_all_models(results, metric='r2_scores')
    
    # Save comparison results
    save_results(
        comparison_results,
        f"{config['output']['dirs']['results']}/model_comparisons.json"
    )
    
    # FDR correction
    logger.info("\nApplying FDR correction...")
    p_values = np.array([comp['p_value'] for comp in comparison_results.values()])
    rejected, corrected_alpha = fdr_correction(
        p_values,
        alpha=config['statistics']['multiple_comparison']['alpha'],
        method=config['statistics']['multiple_comparison']['method']
    )
    logger.info(f"Corrected alpha threshold: {corrected_alpha:.4f}")
    logger.info(f"Significant comparisons: {np.sum(rejected)}/{len(rejected)}")
    
    # =========================================================================
    # STEP 6: VISUALIZATION
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: GENERATING VISUALIZATIONS")
    logger.info("=" * 80)
    
    figure_paths = []
    figures_dir = config['output']['dirs']['figures']
    
    # Performance violin plot
    logger.info("Creating violin plot...")
    fig_path = f"{figures_dir}/performance_violin.png"
    plot_performance_violin(
        results,
        metric='r2_scores',
        output_path=fig_path,
        colors=config['visualization']['colors']
    )
    figure_paths.append(fig_path)
    
    # Performance boxplot
    logger.info("Creating boxplot...")
    fig_path = f"{figures_dir}/performance_boxplot.png"
    plot_performance_boxplot(
        results,
        metric='r2_scores',
        output_path=fig_path
    )
    figure_paths.append(fig_path)
    
    # Comparison heatmap
    logger.info("Creating comparison heatmap...")
    fig_path = f"{figures_dir}/comparison_heatmap.png"
    plot_comparison_heatmap(
        comparison_results,
        metric='p_value',
        output_path=fig_path
    )
    figure_paths.append(fig_path)
    
    # Feature importance plots (for first cognitive score as example)
    if feature_importance_results:
        for conn_type in ['FC', 'SC', 'HC']:
            score_name = score_names[0]
            importance = feature_importance_results[conn_type][score_name]
            
            # Convert to matrix
            n_regions = config['data']['n_regions']
            importance_matrix = feature_vector_to_matrix(
                importance['importance'],
                n_regions=n_regions,
                symmetric=True
            )
            
            # Feature importance matrix
            logger.info(f"Creating feature importance matrix for {conn_type}...")
            fig_path = f"{figures_dir}/importance_{conn_type}_{score_name}.png"
            plot_feature_importance_matrix(
                importance_matrix,
                title=f"{conn_type} Feature Importance: {score_name}",
                output_path=fig_path
            )
            figure_paths.append(fig_path)
    
    logger.info(f"\nGenerated {len(figure_paths)} figures")
    
    # =========================================================================
    # STEP 7: GENERATE REPORT
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: GENERATING HTML REPORT")
    logger.info("=" * 80)
    
    stats_dict = {
        'model_comparisons': comparison_results
    }
    
    report_path = f"{config['output']['dirs']['reports']}/analysis_report.html"
    generate_html_report(
        results_dict=results,
        stats_dict=stats_dict,
        figure_paths=figure_paths,
        config=config,
        output_path=report_path
    )
    
    logger.info(f"HTML report saved to: {report_path}")
    
    # =========================================================================
    # PIPELINE COMPLETE
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE EXECUTION COMPLETE!")
    logger.info("=" * 80)
    logger.info("\nResults Summary:")
    logger.info(f"  - Results saved to: {config['output']['dirs']['results']}")
    logger.info(f"  - Figures saved to: {config['output']['dirs']['figures']}")
    logger.info(f"  - Report available at: {report_path}")
    logger.info("\nTo view the full report, open the HTML file in your web browser.")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
