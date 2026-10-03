import json
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="AI Cardio Risk & Drug Info", page_icon="🫀", layout="wide")
DISCLAIMER = "⚠️ Educational project only. Not a medical diagnosis. Consult a doctor."


@st.cache_resource
def load_model():
    return joblib.load("artifacts/model.pkl"), json.load(open("artifacts/meta.json"))


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


model, meta = load_model()
FEATURES = meta["features"]


def make_row(age, gender, height, weight, ap_hi, ap_lo, chol, gluc, smoke, alco, active):
    bmi = weight / (height / 100) ** 2
    row = dict(gender=gender, height=height, weight=weight, ap_hi=ap_hi, ap_lo=ap_lo,
               cholesterol=chol, gluc=gluc, smoke=smoke, alco=alco, active=active,
               age_years=age, BMI=bmi)
    return pd.DataFrame([row])[FEATURES]   # same column order as training


def risk(df):
    return float(model.predict_proba(df)[0][1])


def level(p):
    return ("LOW", "green") if p < .30 else ("MODERATE", "orange") if p < .60 \
        else ("HIGH", "red") if p < .80 else ("VERY HIGH", "darkred")


# ------------------------------------------------------------------ sidebar
page = st.sidebar.radio("Menu", ["🏠 Home", "🩺 Risk Prediction", "📊 Analytics",
                                 "🤖 Model Comparison", "💊 Drug Information"])
st.sidebar.info(DISCLAIMER)

# ------------------------------------------------------------------ HOME
if page == "🏠 Home":
    st.title("🫀 Cardiovascular Disease Prediction & Drug Information System")
    st.write("Predicts **cardiovascular disease risk** from patient data using Machine "
             "Learning, explains the result, shows data analytics and offers a medicine "
             "information lookup.")
    c1, c2, c3 = st.columns(3)
    best = load_csv("artifacts/metrics.csv").iloc[0]
    c1.metric("Best model", best["Model"])
    c2.metric("Test accuracy", f"{best['Accuracy']*100:.1f}%")
    c3.metric("ROC-AUC", f"{best['ROC_AUC']:.3f}")
    st.warning(DISCLAIMER)

# ------------------------------------------------------------------ PREDICTION
elif page == "🩺 Risk Prediction":
    st.title("🩺 Cardiovascular Risk Prediction")
    with st.form("f"):
        a, b, c = st.columns(3)
        age = a.number_input("Age (years)", 18, 100, 45)
        gender = b.selectbox("Gender", [1, 2], format_func=lambda x: "Female" if x == 1 else "Male")
        height = c.number_input("Height (cm)", 120, 220, 165)
        weight = a.number_input("Weight (kg)", 35.0, 200.0, 70.0)
        ap_hi = b.number_input("Systolic BP (upper)", 70, 250, 120)
        ap_lo = c.number_input("Diastolic BP (lower)", 40, 150, 80)
        chol = a.selectbox("Cholesterol", [1, 2, 3],
                           format_func=lambda x: {1: "Normal", 2: "Above normal", 3: "Well above"}[x])
        gluc = b.selectbox("Glucose", [1, 2, 3],
                           format_func=lambda x: {1: "Normal", 2: "Above normal", 3: "Well above"}[x])
        smoke = c.selectbox("Smoker?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        alco = a.selectbox("Alcohol?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        active = b.selectbox("Physically active?", [1, 0], format_func=lambda x: "Yes" if x else "No")
        go_btn = st.form_submit_button("Predict")

    if go_btn:
        if ap_hi <= ap_lo:
            st.error("Systolic BP must be greater than diastolic BP."); st.stop()
        row = make_row(age, gender, height, weight, ap_hi, ap_lo, chol, gluc, smoke, alco, active)
        p = risk(row)
        lvl, colr = level(p)
        bmi = row["BMI"].iloc[0]

        fig = go.Figure(go.Indicator(mode="gauge+number", value=p * 100,
                        number={"suffix": "%"}, title={"text": f"Risk: {lvl}"},
                        gauge={"axis": {"range": [0, 100]}, "bar": {"color": colr}}))
        fig.update_layout(height=300, margin=dict(t=60, b=0))
        st.plotly_chart(fig, width="stretch")

        # ---- What-if analysis (explainability): risk drop if each factor is improved
        base = dict(age=age, gender=gender, height=height, weight=weight, ap_hi=ap_hi,
                    ap_lo=ap_lo, chol=chol, gluc=gluc, smoke=smoke, alco=alco, active=active)
        ideal_w = round(22 * (height / 100) ** 2, 1)
        what_if = {"Normal BMI": ("weight", ideal_w, bmi >= 25),
                   "Systolic BP 115": ("ap_hi", 115, ap_hi >= 130),
                   "Diastolic BP 75": ("ap_lo", 75, ap_lo >= 85),
                   "Normal cholesterol": ("chol", 1, chol > 1),
                   "Normal glucose": ("gluc", 1, gluc > 1),
                   "Quit smoking": ("smoke", 0, smoke == 1),
                   "No alcohol": ("alco", 0, alco == 1),
                   "Become active": ("active", 1, active == 0)}
        rows = []
        for label, (k, v, applies) in what_if.items():
            if applies:
                new = dict(base); new[k] = v
                if k == "ap_hi" and v <= new["ap_lo"]: continue
                rows.append((label, (p - risk(make_row(**new))) * 100))
        recs = {"Normal BMI": "Maintain healthy weight with balanced diet and exercise.",
                "Systolic BP 115": "Control systolic BP: less salt, regular monitoring.",
                "Diastolic BP 75": "Control diastolic BP; follow doctor's advice.",
                "Normal cholesterol": "Reduce saturated fat; get lipid profile checked.",
                "Normal glucose": "Monitor blood sugar regularly.",
                "Quit smoking": "Quit smoking - biggest reversible risk factor.",
                "No alcohol": "Reduce alcohol consumption.",
                "Become active": "Do at least 30 minutes of physical activity daily."}

        left, right = st.columns(2)
        with left:
            st.subheader("What drives your risk? (what-if)")
            if rows:
                wf = pd.DataFrame(rows, columns=["If you improve", "Risk drops by (%)"]) \
                    .sort_values("Risk drops by (%)", ascending=False)
                st.plotly_chart(px.bar(wf, x="Risk drops by (%)", y="If you improve",
                                       orientation="h"), width="stretch")
            else:
                st.success("No modifiable risk factor found.")
        with right:
            st.subheader("Recommendations")
            for label, _ in rows:
                st.write("✅", recs[label])
            if age >= 50: st.write("✅ Schedule routine cardiovascular check-ups.")
            if not rows and age < 50: st.write("✅ Continue your healthy lifestyle.")

        st.subheader("AI summary")
        top = ", ".join(r[0].lower() for r in sorted(rows, key=lambda r: -r[1])[:3]) or "none"
        st.info(f"The model estimates a **{p*100:.1f}% ({lvl.lower()})** chance of cardiovascular "
                f"disease for a {age}-year-old with BMI {bmi:.1f} and BP {ap_hi}/{ap_lo}. "
                f"Biggest improvable factors: {top}. {DISCLAIMER}")

# ------------------------------------------------------------------ ANALYTICS
elif page == "📊 Analytics":
    st.title("📊 Data Analytics")
    d = load_csv("artifacts/sample.csv")
    d["Disease"] = d["cardio"].map({0: "No", 1: "Yes"})
    a, b = st.columns(2)
    a.plotly_chart(px.pie(d, names="Disease", title="Cardiovascular disease split"), width="stretch")
    b.plotly_chart(px.histogram(d, x="age_years", color="Disease", barmode="overlay",
                                title="Age distribution"), width="stretch")
    a.plotly_chart(px.box(d, x="Disease", y="BMI", title="BMI vs disease"), width="stretch")
    order = ["Normal", "Elevated", "Stage-1", "Stage-2"]
    bp = d.groupby(["bp_category", "Disease"]).size().reset_index(name="count")
    b.plotly_chart(px.bar(bp, x="bp_category", y="count", color="Disease", barmode="group",
                          category_orders={"bp_category": order}, title="BP category vs disease"),
                   width="stretch")
    corr = d.drop(columns=["bp_category", "Disease"]).corr().round(2)
    st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                              title="Correlation heatmap"), width="stretch")

# ------------------------------------------------------------------ MODELS
elif page == "🤖 Model Comparison":
    st.title("🤖 Model Comparison")
    m = load_csv("artifacts/metrics.csv")
    show = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    st.dataframe(m[show].style.format({c: "{:.3f}" for c in show[1:]}), width="stretch")
    st.plotly_chart(px.bar(m.melt(id_vars="Model", value_vars=show[1:]), x="Model", y="value",
                           color="variable", barmode="group"), width="stretch")
    pick = st.selectbox("Confusion matrix of", m["Model"])
    r = m[m["Model"] == pick].iloc[0]
    cm = [[r.TN, r.FP], [r.FN, r.TP]]
    st.plotly_chart(px.imshow(cm, text_auto=True, x=["Pred 0", "Pred 1"], y=["Actual 0", "Actual 1"],
                              color_continuous_scale="Blues"), width="stretch")
    imp = load_csv("artifacts/importance.csv")
    st.plotly_chart(px.bar(imp, x="Importance", y="Feature", orientation="h",
                           title="Feature importance (Random Forest)"), width="stretch")
    st.caption(f"Model used in the prediction page: **{meta['best_model']}** (highest ROC-AUC).")

# ------------------------------------------------------------------ DRUGS
else:
    st.title("💊 Drug Information")
    drugs = load_csv("data/drugs.csv")
    q = st.text_input("Search medicine name (e.g. paracetamol)")
    hits = drugs[drugs["name"].str.contains(q, case=False)] if q else drugs
    if hits.empty:
        st.warning("Medicine not found in the demo database.")
    for _, r in hits.iterrows():
        with st.expander(f"{r['name']}  —  {r['category']}", expanded=bool(q)):
            st.write("**Uses:**", r["uses"])
            st.write("**Side effects:**", r["side_effects"])
            st.write("**Precautions:**", r["precautions"])
            st.write("**Mechanism of action:**", r["mechanism"])
    st.caption(DISCLAIMER)
