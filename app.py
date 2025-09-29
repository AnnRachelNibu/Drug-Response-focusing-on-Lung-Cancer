import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import requests

st.set_page_config(layout="wide")
st.title("Lung Cancer Drug Response Prediction & SHAP Explainability")

# === Load and preprocess dataset ===
@st.cache_data
def load_data():
    df = pd.read_csv("GDSC_DATASET.csv")
    df = df[df["TCGA_DESC"].isin(["LUAD", "LUSC"])]
    df = df.dropna(subset=["TARGET"])
    df["Microsatellite instability Status (MSI)"].fillna(df["Microsatellite instability Status (MSI)"].mode()[0], inplace=True)
    return df

df = load_data()

# === Preprocessing ===
X_raw = df.drop(['LN_IC50', 'COSMIC_ID', 'CELL_LINE_NAME', 'DRUG_ID', 'AUC','Cancer Type (matching TCGA label)'], axis=1)
y = df['LN_IC50']
X = pd.get_dummies(X_raw, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train model ===
@st.cache_resource
def train_model():
    model = XGBRegressor(n_estimators=423, learning_rate=0.1759, max_depth=9,
                         colsample_bytree=0.834, subsample=0.7243, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# === User Inputs ===
st.subheader(" Enter Patient Features")

user_input = {}
for col in X_raw.columns:
    unique_vals = df[col].dropna().unique()
    
    # Skip if only one unique value
    if len(unique_vals) == 1:
        continue

    if df[col].dtype == 'object':
        # Use bubbles (radio buttons) if binary categorical like 'Y'/'N'
        if set(unique_vals) == {"Y", "N"} or set(unique_vals) == {"N", "Y"}:
            user_input[col] = st.radio(f"{col}", unique_vals, horizontal=True)
        else:
            user_input[col] = st.selectbox(f"{col}", unique_vals)
    else:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        user_input[col] = st.slider(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)



input_df = pd.DataFrame([user_input])
input_df = pd.get_dummies(input_df, drop_first=True)

# Ensure the input has all columns model expects
missing_cols = set(X.columns) - set(input_df.columns)
missing_df = pd.DataFrame(0, index=input_df.index, columns=list(missing_cols))
input_df = pd.concat([input_df, missing_df], axis=1)
input_df = input_df[X.columns]

input_df = input_df[X.columns]

# === Prediction ===
prediction = float(model.predict(input_df)[0])
response_type = "drug resistance" if prediction > 4 else "drug sensitivity"

drug_name = user_input.get("DRUG_NAME", "Unknown Drug")  # Fallback if missing

st.subheader(f" Drug: {drug_name}")
st.write(f"**Predicted LN_IC50**: {prediction:.2f} → **{response_type}**")


explainer = shap.Explainer(model)
shap_vals = explainer(input_df)[0]
top_features = pd.Series(shap_vals.values, index=input_df.columns).sort_values(key=abs, ascending=False).head(5)

st.subheader(" Top Contributing Features")
for feat, val in top_features.items():
    st.write(f"- {feat}: {val:.4f}")

st.subheader(" SHAP Waterfall Plot")
fig, ax = plt.subplots()
shap.plots.waterfall(shap_vals, max_display=10, show=False)
st.pyplot(fig)

# === Generate Clinical Summary ===
if st.button("Generate Clinical Summary"):
    feature_text = "\n".join([f"- {feat} ({val:.4f})" for feat, val in top_features.items()])
    prompt_text = f"""
Interpretation and Recommendations for a New Patient

1. Predicted Drug Response
   - The predicted LN_IC50 ({prediction:.2f}) suggests {response_type}.

2. Drug Information
   - Drug Name: {drug_name}

3. Top Contributing Features
{feature_text}

Using this information, please:
- Interpret the drug response,
- Explain the drug’s role (e.g., targeted therapy, cytotoxic agent),
- Discuss metabolism (e.g., liver/kidney involvement),
- Suggest personalized treatment adjustments,
- Recommend 2–3 actionable steps for clinicians.
"""

    headers = {
        "Authorization": f"Bearer {st.secrets['deepseek_api_key']}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek/deepseek-chat:free",
        "messages": [
            {"role": "system", "content": "You are a clinical assistant AI."},
            {"role": "user", "content": prompt_text}
        ]
    }

    with st.spinner("Generating summary..."):
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        if response.status_code == 200:
            summary = response.json()['choices'][0]['message']['content']
            st.subheader(" Clinical Summary")
            st.markdown(summary)
        else:
            st.error(f"API failed: {response.status_code}")
