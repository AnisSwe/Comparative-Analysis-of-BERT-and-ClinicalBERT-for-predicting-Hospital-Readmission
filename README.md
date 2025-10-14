🏥 Comparative Analysis of BERT and ClinicalBERT for Hospital Readmission Prediction

This project compares BERT and ClinicalBERT models for predicting hospital readmission using patient data. The goal is to evaluate how domain-specific pretraining on clinical notes helps improve performance over a general-purpose language model.

📘 Overview

Implemented using Hugging Face Transformers.

Fine-tuned BERT (bert-base-uncased) and ClinicalBERT (emilyalsentzer/Bio_ClinicalBERT) on a hospital readmission dataset.

Compared performance using metrics like Accuracy, F1-score, and AUC.

Found that ClinicalBERT outperforms BERT due to its domain-specific understanding of medical text.

⚙️ Technologies Used

Python

PyTorch

Hugging Face Transformers

Scikit-learn

Pandas, NumPy, Matplotlib
