"""Train cardiovascular-risk models.

Usage:  python train.py data/cardio_train.csv
Outputs (in ./artifacts): model.pkl, metrics.csv, importance.csv,
                          sample.csv (for dashboard), meta.json
"""
import sys, json, os, warnings
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

warnings.filterwarnings("ignore")
SEED = 42
PATH = sys.argv[1] if len(sys.argv) > 1 else "data/cardio_train.csv"
OUT = "artifacts"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- 1. LOAD
df = pd.read_csv(PATH, sep=";")
print("Raw shape:", df.shape)

# ---------------------------------------------------------------- 2. CLEAN
df = df.drop_duplicates()
df["age_years"] = (df["age"] / 365.25).astype(int)
df["BMI"] = df["weight"] / (df["height"] / 100) ** 2

# remove impossible / unrealistic values (medical ranges, NOT IQR, so that
# genuinely high BP patients are kept)
df = df[df["height"].between(120, 220) & df["weight"].between(35, 200)
        & df["ap_hi"].between(70, 250) & df["ap_lo"].between(40, 150)
        & (df["ap_hi"] > df["ap_lo"]) & (df["BMI"] < 60)]


def bp_category(s, d):
    if s < 120 and d < 80: return "Normal"
    if s < 130 and d < 80: return "Elevated"
    if s < 140 and d < 90: return "Stage-1"
    return "Stage-2"


df["bp_category"] = [bp_category(s, d) for s, d in zip(df.ap_hi, df.ap_lo)]
print("Clean shape:", df.shape)

# ---------------------------------------------------------------- 3. FEATURES
FEATURES = ["gender", "height", "weight", "ap_hi", "ap_lo", "cholesterol",
            "gluc", "smoke", "alco", "active", "age_years", "BMI"]
X, y = df[FEATURES], df["cardio"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=SEED, stratify=y)

# ---------------------------------------------------------------- 4. MODELS
def pipe(m):  # scaler + model in ONE object -> no scaling mismatch in the app
    return make_pipeline(StandardScaler(), m)


models = {
    "Logistic Regression": pipe(LogisticRegression(max_iter=1000)),
    "Random Forest": pipe(RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
    "Gradient Boosting": pipe(GradientBoostingClassifier(random_state=SEED)),
    "SVM": pipe(SVC(probability=True, random_state=SEED)),
    "Neural Network (ANN)": pipe(MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True,
        random_state=SEED)),
}

rows, trained = [], {}
for name, m in models.items():
    print(f"Training {name} ...")
    Xf, yf = X_tr, y_tr
    if name == "SVM":  # SVM is slow on big data -> train on 15k sample
        Xf = X_tr.sample(min(15000, len(X_tr)), random_state=SEED)
        yf = y_tr.loc[Xf.index]
    m.fit(Xf, yf)
    pred, prob = m.predict(X_te), m.predict_proba(X_te)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    rows.append(dict(Model=name, Accuracy=accuracy_score(y_te, pred),
                     Precision=precision_score(y_te, pred),
                     Recall=recall_score(y_te, pred),
                     F1=f1_score(y_te, pred),
                     ROC_AUC=roc_auc_score(y_te, prob),
                     TN=tn, FP=fp, FN=fn, TP=tp))
    trained[name] = m
    print(f"   accuracy={rows[-1]['Accuracy']:.4f}  auc={rows[-1]['ROC_AUC']:.4f}")

metrics = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False)
best = metrics.iloc[0]["Model"]
print("\nBEST MODEL:", best)

# ---------------------------------------------------------------- 5. SAVE
joblib.dump(trained[best], f"{OUT}/model.pkl", compress=3)
metrics.to_csv(f"{OUT}/metrics.csv", index=False)
rf = trained["Random Forest"][-1]
pd.DataFrame({"Feature": FEATURES, "Importance": rf.feature_importances_}) \
    .sort_values("Importance", ascending=False) \
    .to_csv(f"{OUT}/importance.csv", index=False)
df.sample(min(8000, len(df)), random_state=SEED)[FEATURES + ["cardio", "bp_category"]] \
    .to_csv(f"{OUT}/sample.csv", index=False)
json.dump({"best_model": best, "features": FEATURES,
           "train_rows": len(X_tr), "test_rows": len(X_te)},
          open(f"{OUT}/meta.json", "w"))
print("Saved everything in ./artifacts")
