# Cardiovascular Disease Prediction & Drug Information System

## 📌 Project Overview

The **Cardiovascular Disease Prediction & Drug Information System** is a machine learning-based healthcare application developed using Python and Streamlit.

The system analyses patient health information and predicts cardiovascular disease risk using machine learning models. It also provides data analytics, model comparison, explainable what-if analysis, and educational drug information in a single web application.

This project demonstrates the practical application of **Data Analytics, Machine Learning, Artificial Intelligence, and Streamlit** in healthcare.

> ⚠️ **Disclaimer:** This is an academic and educational project. It is not a medical diagnostic system and should not be used as a substitute for professional medical advice.

---

## 🎯 Objectives

The main objectives of this project are:

- To collect and preprocess cardiovascular healthcare data.
- To analyse important patient health indicators.
- To perform feature engineering such as Age and BMI calculation.
- To develop machine learning classification models.
- To compare the performance of different machine learning algorithms.
- To predict cardiovascular disease risk from patient information.
- To provide explainable what-if analysis.
- To visualise healthcare data using interactive charts.
- To provide educational drug information.
- To develop a user-friendly web application using Streamlit.
- To deploy the application using Streamlit Community Cloud.

---

## 📊 Dataset

The cardiovascular dataset contains patient health and lifestyle information.

### Major Features

- Age
- Gender
- Height
- Weight
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Cholesterol
- Glucose
- Smoking
- Alcohol Consumption
- Physical Activity
- BMI
- Age in Years

### Target Variable

The target variable is:

- `0` – No cardiovascular disease
- `1` – Cardiovascular disease

---

## 🔄 Methodology

The project follows the following workflow:

```text
Healthcare Dataset
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Exploratory Data Analysis
       ↓
Train-Test Split
       ↓
Machine Learning Models
       ↓
Model Evaluation
       ↓
Best Model Selection
       ↓
Model & Artifacts Saved
       ↓
Streamlit Web Application
       ↓
GitHub Repository
       ↓
Streamlit Cloud Deployment


Data Preprocessing

The following preprocessing steps were performed:

Loaded the cardiovascular dataset.
Removed duplicate records.
Converted age information into years.
Calculated BMI using height and weight.
Checked and filtered unrealistic health measurements.
Created blood pressure categories for analytics.
Separated input features and target variable.
Divided the dataset into training and testing sets.
Applied feature scaling where required.
Used machine learning pipelines for consistent preprocessing and prediction.
🤖 Machine Learning Models

Multiple machine learning algorithms were trained and evaluated.

The implemented models include:

Logistic Regression
Random Forest
Gradient Boosting
Support Vector Machine (SVM)
Neural Network (ANN)
Evaluation Metrics

The models were evaluated using:

Accuracy
Precision
Recall
F1-Score
ROC-AUC
Confusion Matrix
📈 Model Performance

The final test-set results are:

Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
Gradient Boosting	73.3%	75.3%	68.4%	71.7%	0.796
Neural Network (ANN)	73.2%	74.1%	70.5%	72.3%	0.795
Random Forest	72.9%	74.8%	68.4%	71.4%	0.793
Logistic Regression	72.5%	75.3%	66.0%	70.3%	0.787
SVM	72.9%	75.9%	66.1%	70.7%	0.784

The deployed application uses the saved Gradient Boosting model for cardiovascular risk prediction.

🖥️ Application Modules

The Streamlit application contains the following modules:

1. 🏠 Home

The Home page provides:

Project overview
Selected machine learning model
Test accuracy
ROC-AUC score
Educational disclaimer
2. 🩺 Risk Prediction

Users can enter:

Age
Gender
Height
Weight
Systolic BP
Diastolic BP
Cholesterol
Glucose
Smoking status
Alcohol consumption
Physical activity

The trained machine learning model then generates a cardiovascular risk prediction.

3. 🔍 Explainable What-If Analysis

The system provides:

Risk percentage
Important risk-related factors
What-if analysis
Simple recommendations
AI-assisted summary

This helps users understand the model output in a simple way.

4. 📊 Analytics

The Analytics page provides visualisations such as:

Cardiovascular disease distribution
Age distribution
BMI versus disease
Blood pressure category versus disease
5. 🤖 Model Comparison

This module displays:

Accuracy
Precision
Recall
F1-Score
ROC-AUC
Confusion Matrix
Feature importance
6. 💊 Drug Information

Users can search for medicines and view available educational information such as:

Drug name
Drug category
Common uses
Side effects
Precautions
Mechanism of action
🛠️ Technologies Used
Programming Language
Python
Machine Learning
Scikit-learn
Joblib
Data Processing
Pandas
NumPy
Visualisation
Plotly
Matplotlib
Seaborn
Web Application
Streamlit
Development Tools
Visual Studio Code
Jupyter Notebook
Git
GitHub
Deployment
Streamlit Community Cloud

cardio_project/
│
├── artifacts/
│   ├── importance.csv
│   ├── meta.json
│   ├── metrics.csv
│   ├── model.pkl
│   └── sample.csv
│
├── data/
│   ├── cardio_train.csv
│   └── drugs.csv
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore

⚙️ Installation
1. Clone the Repository
git clone https://github.com/Edwinraj07/cardio-project.git
2. Navigate to the Project
cd cardio-project
3. Create a Virtual Environment
python -m venv .venv
4. Activate the Virtual Environment
macOS / Linux
source .venv/bin/activate
Windows
.venv\Scripts\activate
5. Install Dependencies
pip install -r requirements.txt
6. Run the Application
streamlit run app.py

The application will open in the browser.

☁️ Deployment

The application is deployed using Streamlit Community Cloud.

GitHub Repository:

https://github.com/Edwinraj07/cardio-project

Live Application:

https://edwin-cardio-ai.streamlit.app

🔐 Dependency Management

The project uses fixed package versions in requirements.txt to maintain compatibility between the local development environment and the Streamlit Cloud deployment.

streamlit==1.65.0
pandas==2.3.3
numpy==2.2.6
scikit-learn==1.7.2
joblib==1.6.0
plotly==7.1.0

This helps prevent package-version conflicts when loading the saved machine learning model.

📌 Results

The completed system successfully provides:

Cardiovascular disease risk prediction
Machine learning model comparison
Data analytics and visualisation
Explainable what-if analysis
Educational drug information
Interactive Streamlit dashboard
GitHub version control
Cloud deployment

The application was tested locally and successfully deployed on Streamlit Community Cloud.

⚠️ Limitations
The system is intended for academic and educational purposes.
The prediction is based on the available dataset.
Model performance is based on the held-out test dataset.
The prediction should not be considered a medical diagnosis.
Drug information is provided for educational purposes only.
Real-world clinical deployment would require external validation, security, privacy controls, regulatory review, and clinical validation.
🚀 Future Enhancements

Future improvements may include:

External validation using independent datasets.
Advanced hyperparameter tuning.
Additional explainability methods such as SHAP.
Secure user authentication.
Privacy-aware patient data storage.
Database integration.
More comprehensive clinical and medicine references.
Model monitoring and periodic retraining.
👨‍💻 Author

R. Edwin Raj

MSc Data Analytics
Manonmaniam Sundaranar University

📄 License

This project is developed for academic and educational purposes.
