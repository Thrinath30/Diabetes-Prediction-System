# Diabetes Prediction using Binary Classification

## 🔍 Overview
A machine learning project that predicts whether an individual has diabetes based on key medical data. This binary classification project uses the Pima Indians Diabetes dataset and implements multiple supervised learning models to evaluate performance.

## 📊 Problem Statement
- **Goal**: Predict whether a patient has diabetes (1) or not (0) using health metrics.
- **Type**: Binary Classification
- **Target Variable**: `Outcome`

## 📁 Dataset
- **Name**: Pima Indians Diabetes Dataset
- **Source**: [Kaggle / UCI Repository]
- **Features**:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
  - Outcome (Target)

## 🧪 Steps Followed

### 1. Data Understanding
- Loaded data using `pandas`
- Analyzed structure using `.head()`, `.info()`, `.describe()`
- Checked for missing values, outliers, zero values

### 2. Data Cleaning & Preprocessing
- Replaced 0s in `Glucose`, `BMI`, `BloodPressure`, etc. with NaN
- Imputed missing values with median
- Scaled features using `StandardScaler`
- Addressed class imbalance using `Stratified Sampling`

### 3. Exploratory Data Analysis (EDA)
- Histograms and boxplots for distributions
- Violin plots and pairplots for class-wise comparison
- Correlation heatmap

### 4. Feature Engineering
- Created Age groups (e.g., Young, Middle-aged, Senior)
- Removed redundant features
- (Optional) Feature selection with `SelectKBest`

### 5. Data Splitting
- Split into train (80%) and test (20%) using `train_test_split` with stratification

### 6. Model Selection & Training
Tested multiple models:
- Logistic Regression
- Decision Tree
- Random Forest ✅ *(Best performing)*
- K-Nearest Neighbors
- Support Vector Machine
- (Optional) XGBoost

### 7. Model Evaluation
Evaluated with:
- Accuracy
- Confusion Matrix
- Precision, Recall, F1-Score
- ROC Curve & AUC

📌 **Final AUC Score**: 0.8031  
📌 **Final Accuracy**: ~83%

### 8. Hyperparameter Tuning
- Tuned Random Forest using `GridSearchCV`

### 9. Interpretability
- Feature importance from Random Forest
- SHAP values for both global and local explainability (optional but included)

### 10. Final Testing
- Validated best model on unseen test set
- Reported metrics to confirm generalization

## 🧠 Insights
- `Glucose`, `BMI`, and `Age` are the most important predictors of diabetes.
- SHAP plots showed clear relationships between these features and model decisions.

## ⚠️ Limitations
- Small dataset (768 samples)
- Contains only females of Pima Indian heritage (limited generalization)
- Missing values in features like Insulin and SkinThickness

## 🚀 Future Work
- Expand dataset with diverse demographics
- Add lifestyle-related features (diet, activity)
- Deploy model using Streamlit or Flask for demo

## 🛠️ Tools & Libraries
- Python, pandas, numpy
- scikit-learn, matplotlib, seaborn
- SHAP, GridSearchCV

## Author
*DARLA THRINATH*  
*Machine Learning Enthusiast / Data Scientist*
