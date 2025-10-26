# Brain-Connectivity-Based-Cognitive-Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.1007/s00429--021--02289--6-blue)](https://doi.org/10.1007/s00429-021-02289-6)
[![FAU Erlangen-Nürnberg](https://img.shields.io/badge/Affiliation-FAU%20Erlangen--Nürnberg-blue)](https://www.fau.eu/)

Machine learning pipeline to predict crystallized and fluid cognition from functional and structural brain connectivity. Complete Python implementation with 100 train/test splits, permutation testing, and automated HTML reports.

<p align="center">
  <img src="docs/pipeline_overview.png" alt="Pipeline Overview" width="800"/>
</p>

---

## Overview

This system uses **ridge regression** with **nested cross-validation** to predict cognitive scores from:
- **Functional Connectivity (FC)**: Correlation of fMRI BOLD signals between brain regions
- **Structural Connectivity (SC)**: White matter fiber tracts from diffusion MRI
- **Hybrid Connectivity (HC)**: Combination of FC and SC

### Cognitive Measures Predicted
- **Crystallized Cognition**: Language, vocabulary, accumulated knowledge
- **Fluid Cognition**: Processing speed, working memory, executive function
- **Total Cognition**: Overall cognitive ability

---

## Key Features

**Complete Implementation** of Dhamala et al. (2021) methodology  
**Synthetic Data Generation** for immediate usability (no data access barriers)  
**Ridge Regression** with nested cross-validation  
**100 Train/Test Splits** for robust performance estimation  
**Permutation Testing** for statistical significance  
**Feature Importance** extraction using Haufe et al. (2014) activation patterns  
**Publication-Quality Visualizations**  
**Comprehensive HTML Reports**  
**CPU-Optimized** (no GPU required)  
**Modular Architecture** (25 files, 2000+ lines of code)  

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- 8 GB RAM minimum
- ~50 GB disk space (if using real HCP data)

### Installation

```
# 1. Clone the repository
git clone https://github.com/JamilHanouneh/brain-connectivity-prediction.git
cd brain-connectivity-prediction

# 2. Run setup (installs dependencies and creates directories)
python setup_environment.py

# 3. Run the pipeline
python run_pipeline.py --config config.yaml
```

### Quick Test (5 minutes)

```
# Run with reduced iterations for quick testing
python run_pipeline.py --config config.yaml --quick
```

### View Results

Open `outputs/reports/analysis_report.html` in your web browser to see:
- Model performance metrics (R², correlations)
- Statistical significance tests
- Feature importance visualizations
- Model comparisons

---

## Using Your Own Data

### Option 1: Human Connectome Project Data (Recommended)

1. **Register** at [Human Connectome Project](https://www.humanconnectome.org)
2. **Download** the S1200 release:
   - Resting-state fMRI connectivity matrices
   - Diffusion MRI connectivity matrices
   - Behavioral/cognitive scores (`unrestricted_behavioral.csv`)

3. **Organize data** in this structure:
   ```
   data/raw/
   ├── connectivity/
   │   ├── FC/
   │   │   ├── sub-100307_FC.npy
   │   │   ├── sub-100408_FC.npy
   │   │   └── ...
   │   └── SC/
   │       ├── sub-100307_SC.npy
   │       ├── sub-100408_SC.npy
   │       └── ...
   └── behavioral/
       └── cognitive_scores.csv
   ```

4. **Update config.yaml**:
   ```
   data:
     synthetic:
       use_synthetic: false  # Switch to real data
     raw_dir: "data/raw/connectivity"
     behavioral_file: "data/raw/behavioral/cognitive_scores.csv"
   ```

### Option 2: Synthetic Data (No Download Required)

The project includes synthetic data generation that matches HCP statistical properties:

```
data:
  synthetic:
    use_synthetic: true  # Default - works immediately!
```

This generates realistic connectivity matrices and cognitive scores without needing large downloads.

---

## Project Structure

```
brain_cognition_prediction/
├── config.yaml                 # Configuration file (modify parameters here)
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
├── README.md                   # This file
├── setup_environment.py        # Environment setup script
├── run_pipeline.py            # Main execution script
├── data/
│   ├── raw/                   # Raw connectivity matrices (user provides)
│   └── processed/             # Preprocessed data
├── src/
│   ├── data/                  # Data loading and preprocessing
│   │   ├── download.py
│   │   ├── load_connectivity.py
│   │   ├── generate_synthetic.py
│   │   └── preprocess.py
│   ├── models/                # Ridge regression and feature importance
│   │   ├── ridge_prediction.py
│   │   ├── permutation_test.py
│   │   └── feature_importance.py
│   ├── evaluation/            # Performance metrics and comparisons
│   │   ├── metrics.py
│   │   └── compare_models.py
│   ├── visualization/         # Plotting functions
│   │   ├── plot_results.py
│   │   ├── plot_features.py
│   │   └── report_generator.py
│   └── utils/                 # Logging and I/O utilities
│       ├── logger.py
│       └── io.py
├── outputs/
│   ├── models/                # Saved model weights
│   ├── results/               # Performance metrics (JSON/CSV)
│   ├── figures/               # Plots (PNG/SVG)
│   ├── reports/               # HTML reports
│   └── logs/                  # Execution logs
├── notebooks/                 # Jupyter notebooks for exploration
├── tests/                     # Unit tests
└── docs/                      # Documentation and figures
```

---

## Configuration

Edit `config.yaml` to customize:

### Data Parameters
```
data:
  n_subjects: 415              # Number of subjects
  n_regions: 86                # Brain regions (FreeSurfer parcellation)
  connectivity_types:          # Which connectivity types to analyze
    - FC
    - SC
    - HC
  cognitive_targets:           # Which cognitive scores to predict
    - Crystallized
    - Fluid
    - Total
```

### Model Parameters
```
model:
  ridge:
    alpha_range: [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
  train_test:
    n_splits: 100              # Number of random train/test splits
    test_size: 0.2             # 80/20 split
```

### Statistical Testing
```
statistics:
  permutation:
    n_permutations: 1000       # Permutation test iterations
  multiple_comparison:
    method: "fdr_bh"           # FDR correction method
    alpha: 0.05                # Significance level
```

---

## Results

Based on the original paper (Dhamala et al., 2021):

| Model | Cognitive Score | R² Range | Interpretation |
|-------|----------------|----------|----------------|
| FC → Crystallized | 0.06-0.23 | Good | FC predicts language/vocabulary |
| FC → Fluid | 0.15-0.20 | Good | FC predicts processing speed |
| SC → Crystallized | 0.03-0.08 | Moderate | SC weakly predicts language |
| SC → Fluid | 0.05-0.08 | Moderate | SC weakly predicts speed |
| HC → Total | 0.08-0.21 | Good | Combined connectivity helps |

**Key Finding**: Functional connectivity generally outperforms structural connectivity for cognitive prediction.

**Note**: With synthetic data, expect lower R² values (0.02-0.15) as synthetic generation lacks true brain-cognition relationships.

---

## Outputs

### Results Files

- `outputs/results/FC_Crystallized_results.json`: Performance metrics for each model
- `outputs/results/model_comparisons.json`: Statistical comparisons between models
- `outputs/results/FC_Crystallized_importance.pkl`: Feature importance matrices

### Visualizations

- `performance_violin.png`: Distribution of R² scores across models
- `performance_boxplot.png`: Boxplots of performance metrics
- `comparison_heatmap.png`: Pairwise model comparison p-values
- `importance_FC_Crystallized.png`: Feature importance matrices

### HTML Report

Comprehensive report with:
- Configuration summary
- Performance tables
- Statistical test results
- Embedded visualizations
- Interpretation guidance

---

## Running Tests

```
# Run automated tests
python tests/test_project.py

# Test individual components
python tests/test_data.py
python tests/test_model.py
python tests/test_visualization.py
```

---

## Documentation

### Scientific Background

The human brain's cognitive abilities emerge from complex interactions between brain regions. This project tests whether:

1. **Individual differences** in cognition can be predicted from brain connectivity
2. **Different connectivity types** (functional vs. structural) predict different cognitive abilities
3. **Specific brain connections** are more important than others

### Methodology

Following **Dhamala et al. (2021)**:

1. **Functional Connectivity**: Pearson correlation between fMRI time series
   - Fisher z-transformed
   - Upper triangle extracted (symmetric matrix)

2. **Structural Connectivity**: Probabilistic tractography streamline density
   - Log-transformed
   - Normalized to [0, 1]

3. **Hybrid Connectivity**: FC in upper triangle, SC in lower triangle

4. **Prediction Model**: Ridge regression (linear model with L2 regularization)
   - Nested cross-validation for hyperparameter tuning
   - 100 random train/test splits (80/20)

5. **Feature Importance**: Haufe et al. (2014) activation patterns
   - Transforms backward model weights → interpretable forward patterns

---

## Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add support for different brain parcellations (Schaefer, AAL, etc.)
- [ ] Implement additional ML algorithms (SVM, XGBoost, neural networks)
- [ ] Add support for other neuroimaging datasets (UK Biobank, ABCD)
- [ ] Optimize performance with parallel processing
- [ ] Add unit tests and integration tests
- [ ] Create Docker container for easy deployment
- [ ] Add CI/CD pipeline

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Citation

If you use this code in your research, please cite:

### Original Paper
```
@article{dhamala2021distinct,
  title={Distinct functional and structural connections predict crystallised and fluid cognition in healthy adults},
  author={Dhamala, Elvisha and Jamison, Keith W and Jaywant, Abhishek and Dennis, Sarah and Kuceyeski, Amy},
  journal={Brain Structure and Function},
  volume={226},
  pages={1669--1691},
  year={2021},
  publisher={Springer}
}
```

### This Implementation
```
@software{hanouneh2025brain,
  title={Brain Connectivity-Based Cognitive Prediction: A Production-Ready Pipeline},
  author={Hanouneh, Jamil},
  year={2025},
  url={https://github.com/JamilHanouneh/brain-connectivity-prediction},
  affiliation={Friedrich-Alexander-Universität Erlangen-Nürnberg}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Jamil Hanouneh**

- Affiliation: [Friedrich-Alexander-Universität Erlangen-Nürnberg](https://www.fau.eu/)
- GitHub: [@JamilHanouneh](https://github.com/JamilHanouneh)
- Email: jamil.hanouneh1997@gmail.com

---

## Acknowledgments

- Dr. Elvisha Dhamala and colleagues for the foundational methodology
- Human Connectome Project for data standards and best practices
- Friedrich-Alexander-Universität Erlangen-Nürnberg for academic support
- Open-source neuroimaging community (nibabel, nilearn, scikit-learn)

---

## References

### Key Papers

1. **Dhamala et al. (2021)** - Original methodology  
   *Brain Structure and Function*, 226, 1669-1691

2. **Haufe et al. (2014)** - Feature importance method  
   *NeuroImage*, 87, 96-110

3. **Van Essen et al. (2013)** - Human Connectome Project  
   *NeuroImage*, 80, 62-79

4. **Finn et al. (2015)** - Connectome fingerprinting  
   *Nature Neuroscience*, 18(11), 1664-1671

---

## Troubleshooting

### Common Issues

**Problem**: `ModuleNotFoundError`
```
# Solution: Reinstall dependencies
pip install -r requirements.txt
```

**Problem**: Out of memory
```
# Solution: Reduce dataset size in config.yaml
data:
  n_subjects: 100  # Reduce from 415
```

**Problem**: Slow execution
```
# Solution: Use quick mode
python run_pipeline.py --config config.yaml --quick
```

For more help, see [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) or open an issue.

---

## 📊 Project Status

- Core pipeline implementation complete
- Synthetic data generation working
- Statistical validation implemented
- Visualization and reporting functional
- Documentation complete
- Real HCP data integration (user provides)
- Deep learning extensions (planned)
- Clinical population validation (planned)

---

## Star History

If you find this project useful, please consider giving it a star ⭐!

---

<p align="center">
  Made with 🧠 for neuroscience and machine learning education
</p>

<p align="center">
  <sub>Based on the methodology from Dhamala et al. (2021)</sub>
</p>
```
