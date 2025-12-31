# Synthetic vs Real Speech Detection using Cepstral Features

This repository contains the Python code developed in support of an award-winning research study on the detection of synthetic speech through classical machine learning techniques applied to cepstral features.

The work focuses on distinguishing real human voices from text-to-speech (TTS) generated audio by leveraging Gammatone Frequency Cepstral Coefficients (GTCC), their temporal derivatives (ΔGTCC), and a feature selection strategy based on Recursive Feature Elimination (RFE).

---

## Context

Recent advances in speech synthesis have significantly increased the realism of artificial voices, raising concerns in domains such as biometric authentication and digital security.  
The research underlying this repository investigates whether handcrafted acoustic features, combined with lightweight and interpretable machine learning models, can still provide reliable discrimination between real and synthetic speech.

This code supports the study:

**“Advancing Voice Authentication: Insights from Cepstral Coefficients and Recursive Feature Elimination in Speech Signal”**,  
presented at the *12th International Conference on E-Health and Bioengineering (EHB 2024)*, where it received a scientific award.

---

## Methodological overview

The implemented pipeline follows a structured and reproducible workflow:

1. **Audio feature extraction**
   - Gammatone Frequency Cepstral Coefficients (GTCC)
   - Delta Gammatone Frequency Cepstral Coefficients (ΔGTCC)

2. **Dataset preparation**
   - Balanced sampling of real and synthetic speech instances
   - Separation into training, validation, and testing sets

3. **Feature selection**
   - Recursive Feature Elimination (RFE) applied to progressively reduce dimensionality
   - Identification of the most discriminative cepstral coefficients

4. **Classification**
   - Logistic Regression models trained on selected feature subsets
   - Evaluation performed on unseen test data

The analysis highlights the role of ΔGTCC features in capturing temporal dynamics that are harder for synthetic speech to reproduce, leading to improved classification performance with a reduced number of features.

---

## Results summary

The study shows that:
- ΔGTCC features consistently outperform static GTCC features across different feature subset sizes
- A compact subset of five ΔGTCC coefficients achieves competitive performance while maintaining model interpretability
- Feature selection plays a key role in improving robustness and reducing model complexity

The final model achieves approximately 70% accuracy on the test set using only five ΔGTCC features, emphasizing the effectiveness of targeted feature selection in synthetic speech detection tasks.

---

## What is included in this repository

- Python scripts for cepstral feature extraction
- Machine learning code for training and evaluating Logistic Regression models
- Implementation of Recursive Feature Elimination for feature selection
- Experiment-oriented code organization focused on reproducibility

---

## What is not included

- Original audio datasets (not publicly redistributable)
- Dataset-specific configuration files tied to private storage environments

This repository represents a cleaned and public-facing version of the original research code.

---

## Reproducibility considerations

The code is organized to encourage reproducible experimentation through:
- Clear separation between feature extraction, modeling, and evaluation steps
- Deterministic data splits and controlled experimental workflows
- Lightweight dependencies and transparent model choices

---

## Reference

If you use or refer to this code, please cite the associated paper:

M. G. Di Cesare, D. Perpetuini, D. Cardone, A. Merla,  
*Advancing Voice Authentication: Insights from Cepstral Coefficients and Recursive Feature Elimination in Speech Signal*,  
Proceedings of the 12th International Conference on E-Health and Bioengineering (EHB 2024).  
Available at: https://ieeexplore.ieee.org/abstract/document/10805645

---

## License

This project is released under the MIT License.
