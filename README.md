# 🫀 AI-Based Cardiovascular Risk Prediction & Drug Information System

## 1. Project folder

```
cardio_project/
├── data/
│   ├── cardio_train.csv   <- Kaggle-la irundhu download panni inga podunga
│   └── drugs.csv          <- drug information database (already irukku)
├── train.py               <- data clean + 5 models train + best model save
├── app.py                 <- Streamlit web app (UI)
├── requirements.txt       <- deploy-ku thevaiyana libraries
└── artifacts/             <- train.py run panna thaan create aagum (model.pkl etc.)
```

## 2. Enna use panrom, edhuku? (viva-ku ready-a irukkanum)

| Step | Enna panrom | Edhula | Edhuku |
|---|---|---|---|
| Data | `cardio_train.csv` (70,000 patients, Kaggle "Cardiovascular Disease dataset") | pandas | Real patient records vechu train panna |
| Cleaning | duplicate remove, `age` days -> years, BMI = weight/height², impossible values remove (BP, height, weight) | pandas | Thappa data irundha model thappa kathukkum |
| Features | 12 inputs: gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, age_years, BMI | pandas | Model-ku input |
| Split | 80% train / 20% test, `stratify=y` | scikit-learn | Model pudhu data-la eppadi work aagum nu test panna |
| Scaling | `StandardScaler` (Pipeline kulla) | scikit-learn | LR, SVM, ANN-ku values same scale venum |
| Models | Logistic Regression, **Random Forest, SVM, ANN (MLP)**, Gradient Boosting | scikit-learn | Doc-la sonna 3 algorithms + compare panna |
| Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix | scikit-learn | Model quality measure |
| Best model | Highest ROC-AUC model -> `model.pkl` | joblib | Web app load panni use panna |
| Explainability | "What-if" analysis: oru factor (BP, smoking...) improve panna risk evlo kuraiyum | app.py | Risk-ku **en** idhu vandhuchu nu user-ku puriya |
| AI summary | Result-a simple sentence-a generate panrom | app.py | Doc-la sonna "AI-generated summary" |
| Analytics | Pie, histogram, box plot, bar, correlation heatmap | Plotly | EDA charts |
| Drug module | `drugs.csv` search -> uses, side effects, precautions, mechanism | pandas | Doc-la sonna drug information |
| UI | 5 pages web app | Streamlit | Simple-a web app + free deploy |

## 3. Run panra steps (Local - unga laptop)

```bash
# 1. Python 3.10+ install irukkanum. Folder-ku ulla po
cd cardio_project

# 2. (optional) virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# 3. libraries install
pip install -r requirements.txt

# 4. Kaggle-la irundhu cardio_train.csv download panni data/ folder-la podunga

# 5. Model train (2-4 min aagum)
python train.py data/cardio_train.csv

# 6. App run
streamlit run app.py
```
Browser-la `http://localhost:8501` open aagum.

> Laptop slow-a irundha: Kaggle/Colab-la `train.py`-a run panni, `artifacts/` folder-a download panni project-kulla podunga.

## 4. Deploy panra steps (FREE - Streamlit Community Cloud)

1. **GitHub account** create pannunga -> New repository (`cardio-risk-app`, Public).
2. Local-la:
   ```bash
   git init
   git add .
   git commit -m "AI cardio project"
   git branch -M main
   git remote add origin https://github.com/<username>/cardio-risk-app.git
   git push -u origin main
   ```
   **Important:** `artifacts/` folder-um (model.pkl) push aaganum. `data/cardio_train.csv` (~11 MB) venam na `.gitignore`-la podalaam - deploy-ku thevai illa.
3. https://share.streamlit.io -> GitHub-la login -> **Create app** -> repo select -> Branch `main` -> Main file `app.py` -> **Deploy**.
4. 2-3 min-la `https://<name>.streamlit.app` link kedaikum. Adhai doc/PPT-la podalaam.

## 5. Unga original code-la irundha thappugal (naan fix pannathu)

| # | Problem | Fix |
|---|---|---|
| 1 | `label=f"{name} ({auc:.3f}9` -> **SyntaxError** | Syntax sari pannitten; ROC-AUC value ippo metrics table-la varum |
| 2 | `best_model=ensemble`, `best_rf`, `importance` **define pannala** -> NameError | Ellam `train.py`-la proper-a define panniten |
| 3 | `print(contribution[[...]].head(5)])` extra `]` | Remove |
| 4 | `google.colab`, `drive.mount`, `!pip`, `npx localtunnel` | Local/cloud deploy-ku thevai illa, remove |
| 5 | Same model code 2 thadava, `input()` use pannirukkeenga | `input()` web app-la work aagadhu -> Streamlit form |
| 6 | Scaling: sila model scaled data, sila unscaled; app-la mismatch aagum | `Pipeline(StandardScaler + model)` - oru file-la ellam |
| 7 | IQR outlier removal BP > ~170 patients-a delete pannidum | Medical range filter (ap_hi 70-250) use panniten |
| 8 | `age` (days) + `age_years` rendume irundhuchu (duplicate) | `age_years` mattum |
| 9 | SHAP deploy-la heavy, ensemble-ku work aagadhu | What-if explanation (light, work aagum) |
| 10 | Doc-la ANN irukku, code-la illa | `MLPClassifier` add panniten |

## 6. ⚠️ Mukkiyamaana unmai (doc-ai update pannunga)

- **Doc-la Diabetes + Heart + Stroke nu iruku, aana code/dataset cardio mattum.** Rendu option: (a) Doc-la "Cardiovascular disease prediction" nu maathunga (easy, recommend), illana (b) Pima diabetes + stroke dataset download panni same `train.py` pattern-la rendu model add pannalam.
- Doc-la PubChem sonnirukkeenga; naan 12 drugs-oda small `drugs.csv` kudutthirukken. Adhai neengalae extend pannalam (same columns).
- Indha dataset-la accuracy **~72-74%** thaan varum (dataset limit). Idhu normal - 95% varalai nu bayapadadheenga. Viva-la "Recall-um ROC-AUC-um check panrom, accuracy mattum illa" nu sollunga.
- Medical tool illa - app-la disclaimer irukku, adhai remove pannadheenga.

## 7. Viva-ku common questions

- **Evaluation-ku accuracy mattum yen podhadhu?** Medical-la Recall (nijama noi irukkravanga-a miss pannama irukka) mukkiyam.
- **Pipeline yen?** Training-um prediction-um same scaling use aaganum.
- **Best model eppadi select?** Highest ROC-AUC on test data.
- **Overfitting?** Train/test split, `min_samples_leaf`, early stopping.
