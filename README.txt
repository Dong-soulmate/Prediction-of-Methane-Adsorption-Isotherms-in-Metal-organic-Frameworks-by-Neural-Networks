######################
Prediction of Methane Adsorption Isotherms in Metal-organic Frameworks by Neural Networks: Two-dimensional Energy Gradient Feature and Masked Learning Mechanism

## Project Overview
This project aims to predict the adsorption isotherms of methane gas on Metal-Organic Frameworks (MOFs) at different temperatures using machine learning methods. We propose a neural network model with a masking operation to improve prediction accuracy and evaluate the model's generalization ability.

## Workflow

1. **Feature Extraction and Data Preparation**
   - Extract various features of MOFs, including geometric features, energy features, and more.
   - Calculate the adsorption isotherms of methane gas on MOFs at different temperatures using constrained Density Functional Theory (cDFT).

2. **Model Construction**
   - Develop a neural network model.
   - Introduce a masking operation into the network to enhance prediction accuracy.
   - For detailed model implementation, please refer to: `2Dnn_masked_4p.py`

3. **Model Evaluation**
   - Evaluate the prediction accuracy of the model on the test set.
   - Assess the model's generalization ability through cross-dataset testing.

## Data Description
We provide part of the preprocessed raw data:
- `mofdatabase_features_clean_data_4P.csv`: Contains MOF features and related adsorption data.

## Usage Instructions
1. View the model implementation: `2Dnn_masked_4p.py`
2. Use the provided dataset for model training and testing.
3. The dataset can be expanded or model parameters adjusted according to specific needs.

## Contact Information
For any questions or suggestions, please contact the project maintainers.




