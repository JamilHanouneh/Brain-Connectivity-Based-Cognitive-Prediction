"""
HTML report generation for analysis results.

Creates comprehensive HTML reports with embedded figures and tables.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def create_results_summary(results_dict: Dict) -> str:
    """
    Create HTML summary of results.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary of results
    
    Returns
    -------
    str
        HTML string
    """
    html = "<div class='results-summary'>\n"
    html += "<h2>Results Summary</h2>\n"
    html += "<table class='summary-table'>\n"
    html += "<tr><th>Model</th><th>Mean R²</th><th>Std R²</th><th>Mean Correlation</th><th>Std Correlation</th></tr>\n"
    
    for conn_type in results_dict.keys():
        for score_name, results in results_dict[conn_type].items():
            mean_r2 = results.get('mean_r2', np.nan)
            std_r2 = results.get('std_r2', np.nan)
            mean_corr = results.get('mean_correlation', np.nan)
            std_corr = results.get('std_correlation', np.nan)
            
            html += f"<tr>"
            html += f"<td>{conn_type} → {score_name}</td>"
            html += f"<td>{mean_r2:.4f}</td>"
            html += f"<td>{std_r2:.4f}</td>"
            html += f"<td>{mean_corr:.4f}</td>"
            html += f"<td>{std_corr:.4f}</td>"
            html += f"</tr>\n"
    
    html += "</table>\n"
    html += "</div>\n"
    
    return html


def create_figures_section(figure_paths: List[str]) -> str:
    """
    Create HTML section with embedded figures.
    
    Parameters
    ----------
    figure_paths : list of str
        Paths to figure files
    
    Returns
    -------
    str
        HTML string
    """
    html = "<div class='figures-section'>\n"
    html += "<h2>Visualizations</h2>\n"
    
    for fig_path in figure_paths:
        fig_path = Path(fig_path)
        if fig_path.exists():
            fig_name = fig_path.stem.replace('_', ' ').title()
            html += f"<div class='figure'>\n"
            html += f"<h3>{fig_name}</h3>\n"
            html += f"<img src='{fig_path.name}' alt='{fig_name}' style='max-width:100%; height:auto;'>\n"
            html += f"</div>\n"
    
    html += "</div>\n"
    
    return html


def create_statistics_section(stats_dict: Dict) -> str:
    """
    Create HTML section with statistical results.
    
    Parameters
    ----------
    stats_dict : dict
        Dictionary of statistical test results
    
    Returns
    -------
    str
        HTML string
    """
    html = "<div class='statistics-section'>\n"
    html += "<h2>Statistical Analysis</h2>\n"
    
    if 'permutation_tests' in stats_dict:
        html += "<h3>Permutation Test Results</h3>\n"
        html += "<table class='stats-table'>\n"
        html += "<tr><th>Model</th><th>True Score</th><th>p-value</th><th>Z-score</th><th>Significant</th></tr>\n"
        
        for model_name, results in stats_dict['permutation_tests'].items():
            true_score = results.get('true_score', np.nan)
            p_value = results.get('p_value', np.nan)
            z_score = results.get('z_score', np.nan)
            significant = "Yes" if p_value < 0.05 else "No"
            sig_class = "significant" if p_value < 0.05 else "not-significant"
            
            html += f"<tr class='{sig_class}'>"
            html += f"<td>{model_name}</td>"
            html += f"<td>{true_score:.4f}</td>"
            html += f"<td>{p_value:.4f}</td>"
            html += f"<td>{z_score:.2f}</td>"
            html += f"<td><strong>{significant}</strong></td>"
            html += f"</tr>\n"
        
        html += "</table>\n"
    
    if 'model_comparisons' in stats_dict:
        html += "<h3>Model Comparison Results</h3>\n"
        html += "<p>Pairwise comparisons between all models.</p>\n"
        html += "<table class='stats-table'>\n"
        html += "<tr><th>Model 1</th><th>Model 2</th><th>Mean Diff</th><th>p-value</th><th>Cohen's d</th></tr>\n"
        
        for (model1, model2), results in stats_dict['model_comparisons'].items():
            mean_diff = results.get('mean_difference', np.nan)
            p_value = results.get('p_value', np.nan)
            cohens_d = results.get('cohens_d', np.nan)
            
            html += f"<tr>"
            html += f"<td>{model1}</td>"
            html += f"<td>{model2}</td>"
            html += f"<td>{mean_diff:.4f}</td>"
            html += f"<td>{p_value:.4f}</td>"
            html += f"<td>{cohens_d:.3f}</td>"
            html += f"</tr>\n"
        
        html += "</table>\n"
    
    html += "</div>\n"
    
    return html


def generate_html_report(
    results_dict: Dict,
    stats_dict: Optional[Dict] = None,
    figure_paths: Optional[List[str]] = None,
    config: Optional[Dict] = None,
    output_path: str = "outputs/reports/analysis_report.html"
) -> None:
    """
    Generate comprehensive HTML report.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary of model results
    stats_dict : dict, optional
        Dictionary of statistical results
    figure_paths : list of str, optional
        Paths to figure files
    config : dict, optional
        Configuration dictionary
    output_path : str
        Path to save HTML report
    """
    logger.info(f"Generating HTML report: {output_path}")
    
    # Create HTML structure
    html = "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
    html += "<meta charset='UTF-8'>\n"
    html += "<meta name='viewport' content='width=device-width, initial-scale=1.0'>\n"
    html += "<title>Brain Connectivity Prediction - Analysis Report</title>\n"
    
    # Add CSS styles
    html += """
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
        }
        .header p {
            margin: 10px 0 0 0;
            font-size: 1.1em;
            opacity: 0.9;
        }
        .section {
            background: white;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h2 {
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }
        h3 {
            color: #764ba2;
            margin-top: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #667eea;
            color: white;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .significant {
            background-color: #d4edda;
        }
        .not-significant {
            background-color: #f8d7da;
        }
        .figure {
            margin: 30px 0;
            text-align: center;
        }
        .figure img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .footer {
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }
        .info-box {
            background-color: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
        .warning-box {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
            border-radius: 4px;
        }
    </style>
    """
    
    html += "</head>\n<body>\n"
    
    # Header
    html += "<div class='header'>\n"
    html += "<h1>Brain Connectivity-Based Cognitive Prediction</h1>\n"
    html += "<p>Analysis Report</p>\n"
    html += f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n"
    html += "</div>\n"
    
    # Configuration section
    if config is not None:
        html += "<div class='section'>\n"
        html += "<h2>Configuration</h2>\n"
        html += "<div class='info-box'>\n"
        html += f"<p><strong>Number of subjects:</strong> {config['data']['n_subjects']}</p>\n"
        html += f"<p><strong>Number of regions:</strong> {config['data']['n_regions']}</p>\n"
        html += f"<p><strong>Connectivity types:</strong> {', '.join(config['data']['connectivity_types'])}</p>\n"
        html += f"<p><strong>Cognitive targets:</strong> {', '.join(config['data']['cognitive_targets'])}</p>\n"
        html += f"<p><strong>Number of train/test splits:</strong> {config['model']['train_test']['n_splits']}</p>\n"
        html += "</div>\n"
        html += "</div>\n"
    
    # Results summary
    html += "<div class='section'>\n"
    html += create_results_summary(results_dict)
    html += "</div>\n"
    
    # Statistical analysis
    if stats_dict is not None:
        html += "<div class='section'>\n"
        html += create_statistics_section(stats_dict)
        html += "</div>\n"
    
    # Figures
    if figure_paths is not None:
        html += "<div class='section'>\n"
        html += create_figures_section(figure_paths)
        html += "</div>\n"
    
    # Interpretation
    html += "<div class='section'>\n"
    html += "<h2>Interpretation</h2>\n"
    html += "<div class='info-box'>\n"
    html += "<h3>Key Findings:</h3>\n"
    html += "<ul>\n"
    html += "<li><strong>Model Performance:</strong> R² scores indicate the proportion of cognitive score variance explained by connectivity patterns.</li>\n"
    html += "<li><strong>Connectivity Types:</strong> Compare FC (functional), SC (structural), and HC (hybrid) to see which brain connectivity type best predicts cognition.</li>\n"
    html += "<li><strong>Cognitive Abilities:</strong> Different connectivity patterns may predict Crystallized vs. Fluid cognition.</li>\n"
    html += "<li><strong>Statistical Significance:</strong> Permutation tests verify that predictions are better than chance (p < 0.05).</li>\n"
    html += "</ul>\n"
    html += "</div>\n"
    html += "<div class='warning-box'>\n"
    html += "<h3>Notes:</h3>\n"
    html += "<p>This analysis uses ridge regression with nested cross-validation following Dhamala et al. (2021). "
    html += "Results may vary slightly across runs due to random train/test splits. "
    html += "Higher R² and correlation values indicate better predictive performance.</p>\n"
    html += "</div>\n"
    html += "</div>\n"
    
    # References
    html += "<div class='section'>\n"
    html += "<h2>References</h2>\n"
    html += "<p>Dhamala, E., Jamison, K. W., Jaywant, A., Dennis, S., & Kuceyeski, A. (2021). "
    html += "Distinct functional and structural connections predict crystallised and fluid cognition in healthy adults. "
    html += "<em>Brain Structure and Function</em>, 226, 1669-1691.</p>\n"
    html += "<p>Haufe, S., Meinecke, F., Görgen, K., Dähne, S., Haynes, J. D., Blankertz, B., & Bießmann, F. (2014). "
    html += "On the interpretation of weight vectors of linear models in multivariate neuroimaging. "
    html += "<em>NeuroImage</em>, 87, 96-110.</p>\n"
    html += "</div>\n"
    
    # Footer
    html += "<div class='footer'>\n"
    html += "<p>Brain Connectivity-Based Cognitive Prediction Pipeline</p>\n"
    html += "<p>Based on methodology from Dhamala et al. (2021)</p>\n"
    html += "</div>\n"
    
    html += "</body>\n</html>"
    
    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    logger.info(f"HTML report saved to {output_path}")
