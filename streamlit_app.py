import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingClassifier

st.set_page_config(page_title="NAFLD Prediction / Прогноз НАЖБП", page_icon="💉", layout="wide")

MODEL_PATH = "models/ebm_model_v2.pkl"
with open(MODEL_PATH, "rb") as file:
    ebm = pickle.load(file)

selected_features = [feature for feature in ebm.feature_names_in_ if "&" not in feature]

# Language selection
lang = st.radio("🌍 **Select Language / Выберите язык:**", ("English", "Русский"))

# Translations for UI elements
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

st.title(translations["title"][lang])
st.write(translations["desc"][lang])

user_input_dict = {}
for feature in selected_features:
    translated_label = feature_translations[feature][lang]
    user_input_dict[feature] = st.number_input(f"**{translated_label}**", min_value=0.0, max_value=500.0, value=0.0)

if st.button(translations["calculate"][lang]):
    try:
        input_df = pd.DataFrame([user_input_dict])

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
    "Glucose": (3.9, 5.5),
    "Chest Circumference (cm)": (70, 120)
}

# Function to normalize values (0 to 1 scale)
def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5 
    
patient_values = [user_input_dict[feat] for feat in reference_ranges.keys()]
print(patient_values)
min_values = [reference_ranges[feat][0] for feat in reference_ranges.keys()]
max_values = [reference_ranges[feat][1] for feat in reference_ranges.keys()]
normalized_patient_values = [normalize(value, min_val, max_val) for value, min_val, max_val in zip(patient_values, min_values, max_values)]

translated_labels = [feature_translations[feat][lang] for feat in reference_ranges.keys()]

min_scaled_value = min(normalized_patient_values)
max_scaled_value = max(normalized_patient_values)

x_min = min(0, min_scaled_value - 0.2)  
x_max = max(1.1, max_scaled_value + 0.2)  

fig, ax = plt.subplots(figsize=(10, 8))

for i in range(len(reference_ranges)):
    ax.barh(i, 1, left=0, color="gray", alpha=0.4, height=0.5, label=("Normal Range" if lang == "English" else "Норма") if i == 0 else "")

for i, value in enumerate(normalized_patient_values):
    color = "blue" if 0 <= value <= 1 else "red"  # red for extreme values
    ax.scatter(value, i, color=color, s=100, label=("Your Value" if lang == "English" else "Ваше значение") if i == 0 else "")

ax.set_yticks(range(len(reference_ranges)))
ax.set_yticklabels(translated_labels, fontsize=11)
ax.set_xlabel("Normalized Value" if lang == "English" else "Нормализованное значение")
ax.set_title("Comparison of Your Values with Normal Ranges" if lang == "English" else "Сравнение Ваших значений с нормой", fontsize=14, fontweight="bold")
ax.legend()
ax.set_xlim([x_min, x_max])  

st.pyplot(fig)
