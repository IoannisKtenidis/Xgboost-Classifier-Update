🎯 Multiclass Network Traffic Classification with XGBoost and MLP
This repository contains two implementations for classifying types of network traffic (e.g., Gaming, Streaming, Browsing) using:

XGBoost (with Scikit-learn pipeline and evaluation plots)
MLP (Multilayer Perceptron) in PyTorch (with advanced training techniques)

📁 Files

xgboost_multiclass_classifier.py: XGBoost-based classifier with full evaluation and export.
mlp_multiclass_classifier.py: PyTorch-based MLP classifier with Focal Loss, EMA, and more.
Multiclassification_Database.csv: Dataset used for training and evaluation (not included here).


🧪 Dataset
The dataset contains 15 features per sample and labels for 3 traffic categories:

Gaming
Streaming (e.g., TikTok)
Browsing

Columns Timestamp and DRB.RlcDelayUl are excluded from training.

⚙️ XGBoost Pipeline

Preprocessing: StandardScaler
Model: XGBClassifier with multi:softprob objective
Hyperparameter Tuning: GridSearchCV with 5-fold CV and f1_macro scoring
Evaluation:

Accuracy, Precision, Recall, F1 Score
Normalized Confusion Matrix
ROC and Precision–Recall Curves
Feature Importance


Artifacts:

Saved pipeline: xgb_best_pipeline.pkl
Metrics in CSV format
Plots in PNG/PDF




🧠 MLP Classifier (PyTorch)

Architecture: 15 → 16 → Dropout(0.3) → 8 → 3
Loss Function: Focal Loss (instead of CrossEntropy)
Training Enhancements:

Gaussian input noise
Exponential Moving Average (EMA)
CosineAnnealingWarmRestarts for LR scheduling
Early stopping based on validation loss


Evaluation:

Accuracy, Precision, Recall, F1 Score
Confusion Matrix
ROC & PR Curves
Training Curves (Loss & Accuracy)




📊 Requirements

Python 3.8+
PyTorch
Scikit-learn
XGBoost
Matplotlib, Seaborn, Pandas


🚀 Run Instructions
Shell
''
# Run XGBoost classifier
python xgboost_multiclass_classifier.py
# Run MLP classifier
python mlp_multiclass_classifier.pyΕμφάνιση περισσότερων γραμμών
''

📌 Notes

Scripts are compatible with Google Colab.
Fonts and plots are optimized for publication (Times New Roman, PDF export).
The MLP implementation trains multiple seeds and selects the best-performing model.
