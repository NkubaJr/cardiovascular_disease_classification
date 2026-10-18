# Cardiovascular Disease Classification: Traditional Machine Learning vs Deep Learning

Summative assignment for the Machine Learning module, African Leadership University.
This project predicts the presence of cardiovascular disease from routine clinical and lifestyle measurements, and compares scikit-learn models with Keras neural networks under a single, controlled evaluation protocol.


## Contents
1. [Project links](#project-links)
2. [Problem and dataset](#problem-and-dataset)
3. [Method](#method)
4. [Experiments](#experiments)
5. [Final results](#final-results)
6. [Key findings](#key-findings)
7. [Limitations](#limitations)
8. [Repository structure](#repository-structure)
9. [Reproducing the results](#reproducing-the-results)
10. [Data source and references](#data-source-and-references)

## Project links
| Resource | Link |
|---|---|
| Written report | Submitted as PDF on Canvas |
| Google Drive folder (zipped raw dataset, best model weights, experiment logs) | YOUR_DRIVE_LINK |
| Video presentation | YOUR_VIDEO_LINK |

## Problem and dataset
**Task:** binary classification of cardiovascular disease (`cardio`: 1 = disease present, 0 = absent).

**Data:** the public Cardiovascular Disease dataset on Kaggle, 70,000 records and 11 predictors (age, gender, height, weight, systolic and diastolic blood pressure, cholesterol, glucose, smoking, alcohol use, physical activity). The raw data is not stored in this repository. The notebook downloads it with `kagglehub`, and a zipped copy is in the Drive folder above.

**Cleaning:** age converted from days to years; duplicate rows removed; records excluded for implausible values (systolic outside 70-250 mmHg, diastolic outside 40-150 mmHg, diastolic not lower than systolic, height outside 120-220 cm, weight outside 30-200 kg). This removed 2,064 records and left 67,936 (50.3% negative, 49.7% positive).

**Split:** stratified 70/15/15 into 47,555 training, 10,190 validation and 10,191 test records, random seed 42. The scaler is fitted on the training split only. The test set is used once, to evaluate the two selected models.

## Method
- **Feature engineering:** BMI, pulse pressure, mean arterial pressure and a four-level blood-pressure stage.
- **Traditional ML (scikit-learn):** logistic regression, random forest and histogram gradient boosting, with randomised and grid search (3-fold cross-validation, ROC-AUC objective) on the training split.
- **Deep learning (TensorFlow/Keras):** Sequential and Functional API networks fed by a `tf.data` pipeline, with dropout, batch normalisation, L2 regularisation, early stopping and learning-rate scheduling.
- **Evaluation:** accuracy, precision, recall, F1, ROC-AUC, confusion matrices, ROC and precision-recall curves, learning curves, threshold analysis and permutation importance.
- **Experiment tracking:** Weights & Biases (offline mode) and TensorBoard, with logs archived in the Drive folder.

## Experiments
Validation results for all twelve experiments (all use the same split and seed).

| ID | Model | Val. AUC | Val. accuracy | Train-val AUC gap |
|---|---|---|---|---|
| E1 | Logistic regression (base features) | 0.7852 | 0.7213 | 0.0076 |
| E2 | Logistic regression (+ engineered features) | 0.7844 | 0.7226 | 0.0078 |
| E3 | Random forest (default) | 0.7670 | 0.7081 | 0.2330 |
| E4 | Histogram gradient boosting (default) | 0.7954 | 0.7294 | 0.0218 |
| E5 | Random forest (tuned) | 0.7943 | 0.7296 | 0.0260 |
| E6 | Histogram gradient boosting (tuned) | 0.7949 | 0.7291 | 0.0209 |
| E7 | Logistic regression (tuned) | 0.7844 | 0.7228 | 0.0078 |
| E8 | Sequential network 64-32 | 0.7935 | 0.7266 | 0.0123 |
| E9 | Sequential network 128-64-32 + dropout | 0.7937 | 0.7251 | 0.0100 |
| E10 | Functional network, batch norm + L2 | 0.7954 | 0.7275 | 0.0104 |
| E11 | E10 with learning rate 0.05 | 0.7921 | 0.7222 | 0.0069 |
| E12 | E10 with learning rate 3e-4 + scheduler | 0.7947 | 0.7282 | 0.0111 |

## Final results
Held-out test set (10,191 records), models selected by validation AUC.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Histogram gradient boosting (E4) | 0.7335 | 0.7438 | 0.7080 | 0.7254 | 0.7966 |
| Functional neural network (E10) | 0.7341 | 0.7550 | 0.6888 | 0.7204 | 0.7966 |

## Key findings
- Neither paradigm showed a meaningful advantage: both selected models reach a test ROC-AUC of 0.797.
- The default random forest overfitted (training AUC 1.00, validation AUC 0.767). Limiting depth and leaf size removed most of the gap.
- Engineered features did not improve logistic regression, and hyperparameter tuning did not beat default gradient boosting, which suggests the predictors, not the models, limit performance.
- A learning rate of 0.05 (E11) produced unstable validation loss, while a small rate with a scheduler (E12) trained smoothly without improving AUC.
- Systolic blood pressure dominates permutation importance (AUC drop of about 0.16), well ahead of age and cholesterol.
- Lowering the decision threshold from 0.5 to 0.4 raises recall from 0.708 to 0.790 at a cost in precision (0.744 to 0.692), which is relevant for screening.

## Limitations
The results come from one public dataset and one random split, without external validation, calibration or subgroup analysis. The dataset omits clinical history and laboratory values, and its target is a recorded disease label rather than a future outcome. These models are a research exercise and are not a diagnostic tool.

## Repository structure
```text
cardiovascular-disease-classification/
|-- cardiovascular_disease_classification.ipynb   # full pipeline: data, experiments, evaluation, error analysis
|-- requirements.txt                              # Python dependencies
|-- README.md
```
The dataset and trained models are intentionally not committed; they are in the Drive folder linked above.

## Reproducing the results
1. Open the notebook in Google Colab with the badge above, or download it and run it in Jupyter.
2. Run all cells from top to bottom (Runtime, then Run all). The notebook installs its dependencies and downloads the dataset from Kaggle, or prompts you to upload `cardio_train.csv` if the download fails.
3. Expect about 20 minutes on Colab. All seeds are fixed (42), so metrics reproduce closely, although neural-network results can differ slightly across hardware and library versions.

To run locally: `pip install -r requirements.txt`, then open the notebook.

## Data source and references
- S. Ulianova, "Cardiovascular Disease Dataset," Kaggle. https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
- Libraries: scikit-learn, TensorFlow/Keras, pandas, NumPy, matplotlib, seaborn, Weights & Biases, TensorBoard.
- Full literature review and IEEE reference list: see the written report.
