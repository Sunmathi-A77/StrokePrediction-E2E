## 🩺 Stroke Prediction App

### Project Overview

The Stroke Prediction App is a web-based tool built with Python and Streamlit to predict the likelihood of a patient having a stroke. The app uses a Random Forest Classifier trained on healthcare-related features, including demographics, medical history, lifestyle, and lab values.

The app is designed to be user-friendly, with interactive input fields, a colorful and modern interface, and a clear prediction result including risk probability.

### Live App Link: 

🩺 Stroke Prediction App - https://strokeprediction-e2e.streamlit.app/

<img width="700" height="700" alt="Screenshot 2025-10-26 230434" src="https://github.com/user-attachments/assets/8accb1a1-45b4-451d-8c9f-eb347421440a" />

### Dataset

Source: Healthcare Stroke Dataset - https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

Number of records: 5110

#### Features include:

| Column Name       | Description                           | Values / Type                              |
| ----------------- | ------------------------------------- | ------------------------------------------ |
| id                | Patient ID (removed in preprocessing) | Integer                                    |
| gender            | Patient gender                        | Male, Female, Other                        |
| age               | Age in years                          | Numeric                                    |
| hypertension      | Hypertension status                   | 0 = No, 1 = Yes                            |
| heart_disease     | Heart disease status                  | 0 = No, 1 = Yes                            |
| ever_married      | Marriage status                       | Yes, No                                    |
| work_type         | Type of work                          | Private, Self-employed, Govt_job, children |
| Residence_type    | Type of residence                     | Urban, Rural                               |
| avg_glucose_level | Average glucose level                 | Numeric                                    |
| bmi               | Body Mass Index                       | Numeric                                    |
| smoking_status    | Smoking habit                         | formerly smoked, never smoked, smokes      |
| stroke            | Target variable                       | 0 = No, 1 = Yes                            |

### Preprocessing Steps

#### Missing Value Handling

Filled missing bmi values with median

#### Encoding Categorical Variables

gender, ever_married, Residence_type → Label Encoding

work_type and smoking_status → One-Hot Encoding

#### Feature Scaling

StandardScaler applied after SMOTE

#### Skew Correction

Log-transform applied to avg_glucose_level and bmi

#### Outlier Handling

Clipped extreme values for avg_glucose_level and bmi

#### Balancing the Dataset

SMOTE used to handle class imbalance for stroke

### Modeling

#### Models Tested:

Logistic Regression

Random Forest Classifier ✅ (Final)

Gradient Boosting

k-Nearest Neighbors

Support Vector Machine

Naive Bayes

#### Performance:

Random Forest achieved the best accuracy and stability.

Features Importance calculated and visualized.

#### Saved Files:

random_forest_stroke_model.pkl – Trained model

scaler.pkl – StandardScaler

feature_columns.pkl – Columns used for model input

### App Features

Interactive sidebar for patient data input

Log-transform and feature scaling handled automatically

One-hot encoding applied internally

Predicts stroke risk with probability

Visually appealing interface:

Background color

Prediction box with color

Footer credit

### Installation

Clone the repository:
```
git clone https://github.com/yourusername/StrokePredictionApp.git
cd StrokePredictionApp
```

Create a virtual environment and activate it:

```
python -m venv myvenv
# Windows
myvenv\Scripts\activate
# Linux/Mac
source myvenv/bin/activate
```

Install required libraries:
```
pip install -r requirements.txt
```

requirements.txt example:
```
streamlit
pandas
numpy
scikit-learn
imblearn
matplotlib
seaborn
```

### Usage

Run the Streamlit app:

```streamlit run app.py```

Input patient details in the sidebar.

Click Predict Stroke Risk to view prediction and probability.

### Technologies & Libraries

Python 3.x

Streamlit – Web app framework

Pandas & NumPy – Data handling

Scikit-learn – Modeling & preprocessing

Imbalanced-learn – SMOTE

Matplotlib & Seaborn – EDA & visualization
