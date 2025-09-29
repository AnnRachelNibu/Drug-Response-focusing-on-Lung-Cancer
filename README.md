# Drug-Response-focusing-on-Lung-Cancer
This aims to contribute to the field of oncology through the fusion of artificial intelligence with cancer genomics . The study addresses the challenges of tumor heterogeneity and treatment resistance by focusing on LUAD and LUSC, thus helping advance lung cancer treatment and bring about better and more personalized treatment options.
This provides the methodology adopted in this work for predicting drug sensitivity in lung cancer patients for two subtypes of non-small cell lung cancer, i.e., lung adenocarcinoma (LUAD), and lung squamous cell carcinoma (LUSC) with the help of explainable AI and machine learning.
### 1. Data Acquisition and Exploration:
The dataset acquired from Genomics of Drug Sensitivity in Cancer is loaded into a Pandas DataFrame. The dataset is filtered to only contain LUAD and LUSC cancer types. The target variable, LN-IC50, represents the drug sensitivity (a lower LN-IC50 value indicates higher sensitivity). Using matplotlib, seaborn and plotly, exploratory data analysis is performed to understand the correlation within the dataset.
### 2•	Data Preprocessing:
The missing values of variables greater than 5 percent are dropped and those under are filled in with the model values or mean values. One-hot encoding is performed to convert categorical values into numerical values so that machine learning model can understand.
### 3•	Model Training: 
A regressor XGBoost is trained on the selected features with LN-IC50 as target. Hyperparameter search is carried out using Randomized Search. The model is evaluated using RMSE, MAE, and R² on the test set.
### 4•	Explainability:
SHAP is used to explain which features contribute most to the drug sensitivity prediction for each sample. The SHAP values are then passed on to the DeepSeek API, which summarizes them into simple clinical insights, making the results more interpretable and easier to use in personalized treatment planning.
