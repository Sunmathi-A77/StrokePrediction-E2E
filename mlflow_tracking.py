import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from warnings import filterwarnings
import seaborn as sns
import matplotlib.pyplot as plt

filterwarnings("ignore")

df = pd.read_csv("healthcare-stroke-dataset.csv")

df.drop(columns=["id"], inplace=True)

df["bmi"] = df["bmi"].fillna(df["bmi"].median())

df["gender"] = df["gender"].map({"Male": 1, "Female": 0, "Other": 0})
df["ever_married"] = df["ever_married"].map({"Yes": 1, "No": 0})
df["Residence_type"] = df["Residence_type"].map({"Urban": 1, "Rural": 0})
df = pd.get_dummies(df, columns=["smoking_status", "work_type"], drop_first=True)

df['avg_glucose_level'] = np.log1p(df['avg_glucose_level'])
df['bmi'] = np.log1p(df['bmi'])

cols = ['avg_glucose_level', 'bmi']
for col in cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)


X = df.drop("stroke", axis=1)
y = df["stroke"]

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

scaler = StandardScaler()
X_res_scaled = scaler.fit_transform(X_res)

X_train, X_test, y_train, y_test = train_test_split(X_res_scaled, y_res, test_size=0.2, random_state=42, stratify=y_res)

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)


mlflow.set_experiment("Stroke Prediction - Accuracy Based")

models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200, max_depth=20),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "k-NN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=42),
    "Naive Bayes": GaussianNB(),
}

best_acc = 0
best_model = None
best_model_name = None
best_run_id = None


for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros_like(y_pred)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_params({
            "model": name,
            "scaler": "StandardScaler",
            "sampling": "SMOTE",
            "skew_fix": "log1p(avg_glucose_level, bmi)",
            "outlier_clip": "IQR(avg_glucose_level, bmi)"
        })
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "auc": auc
        })

        input_example = pd.DataFrame(X_test[:1], columns=X.columns)
        mlflow.sklearn.log_model(model, name=name, input_example=input_example)

        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}\n")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_model_name = name
            best_run_id = run.info.run_id


if best_model is not None and best_run_id:
    model_uri = f"runs:/{best_run_id}/{best_model_name}"
    mlflow.register_model(model_uri=model_uri, name="Stroke_Prediction_Best_Model")

    with open("random_forest_stroke_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("feature_columns.pkl", "wb") as f:
        pickle.dump(X.columns.tolist(), f)

    print(f"🏆 Best model: {best_model_name} (Accuracy = {best_acc:.4f}) registered & saved locally.")
else:
    print("⚠️ No best model found or MLflow run ID missing.")