# FORL-Net: Training and Evaluation

This package provides the PyTorch implementation of FORL-Net for predicting corporate digital transformation using financial and organizational indicators. The implementation uses a dataset containing 2,000 enterprise records and supports model training, cross-validation, performance evaluation, ablation analysis, and visualization.

## 1. Installation and Execution

Install the required Python libraries:

```bash
python -m pip install -r requirements.txt
```

Run the training and evaluation:

```bash
python train.py --data digital_innovation_dataset_class.csv --out results
```

Run the complete experimental evaluation, including component and relationship analysis:

```bash
python train.py --data digital_innovation_dataset_class.csv --out full_results --variants --branches
```

## 2. Model Implementation

FORL-Net incorporates the following components:

- Feature embedding and category-specific representation learning.
- Within-category relationship learning.
- Cross-category relationship modeling through seven predefined relationship branches.
- Adaptive relationship reliability weighting.
- Financial anchor representation and gating mechanism.
- Residual feature refinement and classification.

The framework processes 23 enterprise predictors organized into seven indicator categories. The `digitaltrans_class` variable serves as the classification target, while `digitaltransindex` is excluded from the input features.

## 3. Training Configuration

The model uses the following training settings:

| Parameter | Configuration |
|---|---|
| Embedding dimension | 64 |
| Relationship hidden dimension | 64 |
| Dropout | 0.25 |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Maximum epochs | 100 |
| Early stopping patience | 12 |
| Random seed | 42 |
| Cross-validation | Stratified five-fold |
| Validation proportion | 20% of each development fold |
| Learning rate scheduling | Cosine annealing |
| Loss function | Class-balanced cross-entropy |

Missing-value imputation and feature standardization are fitted using training data within each fold.

## 4. Performance Evaluation

The evaluation includes accuracy, precision, recall, F1-score, Matthews correlation coefficient (MCC), quadratic weighted kappa (QWK), ROC-AUC, PR-AUC, Brier score, expected calibration error (ECE), and ordinal mean absolute error.

The implementation calculates fold-wise and overall out-of-fold performance, class-wise classification results, and confusion matrices.

## 5. Ablation and Relationship Analysis

The experimental evaluation examines the contributions of the main architectural components through individual component-removal experiments.

Relationship analysis includes the evaluation of seven predefined cross-category branches, comparison of adaptive and equal relationship weighting, relationship attention distributions, and financial gating behavior.

## 6. Results and Visualization

The implementation generates the following experimental outputs:

- Overall and fold-wise performance tables.
- Class-wise precision, recall, and F1-score.
- Confusion matrices.
- ROC and precision–recall curves.
- Calibration and prediction-confidence plots.
- Training and validation loss curves.
- Component and relationship ablation results.
- Relationship attention and financial gating visualizations.
- Training time and model parameter information.


The trained model checkpoints and experimental configuration are also retained to support reproducibility.

## 7. Reproducibility

The implementation uses fixed random seeds, stratified cross-validation, training-only preprocessing, validation-based early stopping, and independent test-fold evaluation.