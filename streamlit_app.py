import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingClassifier

# 🚀 Set up Streamlit page config
st.set_page_config(page_title="NAFLD Prediction / Прогноз НАЖБП", page_icon="💉", layout="wide")

# 🔹 Load the trained EBM model
MODEL_PATH = "models/ebm_model_v2.pkl"
with open(MODEL_PATH, "rb") as file:
    ebm = pickle.load(file)

# 🔹 Extract original features (EBM will create interactions internally)
selected_features = [feature for feature in ebm.feature_names_in_ if "&" not in feature]

# 🌍 Language selection
lang = st.radio("🌍 **Select Language / Выберите язык:**", ("English", "Русский"))

# 🌍 Translations for UI elements
translations = {
    "title": {"English": "NAFLD Prediction", "Русский": "Прогноз НАЖБП"},
    "desc": {
        "English": "Enter your health data to predict the probability of NAFLD.",
        "Русский": "Введите значения показателей для получения прогноза вероятности НАЖБП.",
    },
    "calculate": {"English": "Calculate Prediction", "Русский": "Рассчитать Прогноз"},
    "probability": {"English": "Probability", "Русский": "Вероятность"},
    "class": {"English": "Class", "Русский": "Класс"},
    "healthy": {"English": "Healthy", "Русский": "Здоров"},
    "sick": {"English": "Sick", "Русский": "Болен"},
    "alert": {"English": "🚨 Model predicts that you are sick.", "Русский": "🚨 Модель предсказывает, что вы больны."},
    "success": {"English": "✅ Model predicts that you are healthy.", "Русский": "✅ Модель предсказывает, что вы здоровы."},
    "prediction results": {"English": "Prediction Results", "Русский": "Результаты Прогноза "}
}

# 📌 Feature Translations (for user input fields)
feature_translations = {
    "Visceral Fat (%)": {"English": "Visceral Fat (%)", "Русский": "Висцеральный жир (%)"},
    "ALT": {"English": "ALT", "Русский": "АЛТ"},
    "AST": {"English": "AST", "Русский": "АСТ"},
    "BMI": {"English": "BMI", "Русский": "ИМТ"},
    "GGT": {"English": "GGT", "Русский": "ГГТП"},
    "Chest Circumference (cm)": {"English": "Chest Circumference (cm)", "Русский": "Обхват груди (см)"},
    "CRP": {"English": "CRP", "Русский": "СРБ"},
    "Body Fat (%)": {"English": "Body Fat (%)", "Русский": "Жир (%)"},
    "LDL": {"English": "LDL", "Русский": "ЛПНП"},
    "Ferritin": {"English": "Ferritin", "Русский": "Ферритин"},
    "Skeleton (%)": {"English": "Skeleton (%)", "Русский": "Скелет (%)"},
    "Triglycerides": {"English": "Triglycerides", "Русский": "Триглицериды"},
    "Insulin": {"English": "Insulin", "Русский": "Инсулин"},
    "Glucose": {"English": "Glucose", "Русский": "Глюкоза"}
}

# 🎯 **User Input Form**
st.title(translations["title"][lang])
st.write(translations["desc"][lang])

# 🔹 Collect user inputs dynamically based on selected features
user_input_dict = {}
for feature in selected_features:
    translated_label = feature_translations[feature][lang]
    user_input_dict[feature] = st.number_input(f"**{translated_label}**", min_value=0.0, max_value=500.0, value=20.0)

# 🎯 **Prediction Button**
if st.button(translations["calculate"][lang]):
    try:
        # Convert input dictionary to DataFrame
        input_df = pd.DataFrame([user_input_dict])

        # Ensure correct feature order for the model
        input_df = input_df[selected_features]

        # Predict NAFLD probability
        probability = ebm.predict_proba(input_df)[0][1]
        predicted_class = translations["sick"][lang] if probability >= 0.5 else translations["healthy"][lang]

        # Show results
        st.subheader(translations["prediction results"][lang])
        st.write(f"**{translations['probability'][lang]}:** {probability:.4f}")
        st.write(f"**{translations['class'][lang]}:** {predicted_class}")

        if predicted_class == translations["sick"][lang]:
            st.error(translations["alert"][lang])
        else:
            st.success(translations["success"][lang])

    except Exception as e:
        st.error(f"🚨 Error: {e}")

# 📌 **Feature Importance & Patient Value Visualization**
st.subheader("📊 NAFLD Risk Markers – Normal Ranges vs. Your Values")

# Define reference ranges for key biomarkers (adjust values as needed)
reference_ranges = {
    "Visceral Fat (%)": (5, 15),
    "ALT": (7, 41),
    "AST": (10, 40),
    "GGT": (10, 70),
    "BMI": (18.5, 24.9),
    "CRP": (0.0, 5.0),
    "Body Fat (%)": (10, 25),
    "LDL": (0, 3.0),
    "Ferritin": (10, 250),
    "Skeleton (%)": (30, 40),
    "Triglycerides": (0, 1.7),
    "Insulin": (2.0, 26.0),
    "Glucose": (3.9, 5.5)
}

# Convert patient input into a list
patient_values = [user_input_dict[feat] for feat in reference_ranges.keys()]
min_values = [reference_ranges[feat][0] for feat in reference_ranges.keys()]
max_values = [reference_ranges[feat][1] for feat in reference_ranges.keys()]

# 📊 Plot reference ranges and patient values
fig, ax = plt.subplots(figsize=(10, 8))

# Plot normal range bars
for i, (min_val, max_val) in enumerate(zip(min_values, max_values)):
    ax.barh(i, max_val - min_val, left=min_val, color="gray", alpha=0.4, height=0.5, label="Normal Range" if i == 0 else "")

# Plot patient values as blue dots
ax.scatter(patient_values, range(len(reference_ranges)), color="blue", s=100, label="Your Value")

# Format chart
ax.set_yticks(range(len(reference_ranges)))
ax.set_yticklabels(list(reference_ranges.keys()), fontsize=11)
ax.set_xlabel("Value")
ax.set_title("Comparison of Your Values with Normal Ranges", fontsize=14, fontweight="bold")
ax.legend()
ax.set_xlim([0, max(max_values) * 1.1])  # Slightly extend x-axis

st.pyplot(fig)
