import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier

st.set_page_config(page_title="NAFLD Prediction / Прогноз НАЖБП", page_icon="💉", layout="wide")

# Load the trained EBM model
MODEL_PATH = "models/ebm_model.pkl"
FEATURES_PATH = "data/feature_min_max_values.csv"

with open(MODEL_PATH, "rb") as file:
    ebm = pickle.load(file)
    
# Load feature min/max values from CSV
feature_min_max_df = pd.read_csv(FEATURES_PATH)
feature_min_max_df["Feature"] = feature_min_max_df["Feature"].str.strip().str.replace('"', '')
feature_min = feature_min_max_df.set_index("Feature")["Min"].to_dict()
feature_max = feature_min_max_df.set_index("Feature")["Max"].to_dict()

# Min-Max Scaler function
def min_max_scaler(value, feature_name):
    min_val = feature_min.get(feature_name, 0)  # Default to 0 if missing
    max_val = feature_max.get(feature_name, 1)  # Default to 1 if missing
    return (value - min_val) / (max_val - min_val) if max_val != min_val else 0

# Language Selection
lang = st.radio("🌍 **Select Language / Выберите язык:**", ("English", "Русский"))

# Translations
translations = {
    "title": {"English": "NAFLD Prediction", "Русский": "Прогноз НАЖБП"},
    "desc": {
        "English": "Enter your health data to predict the probability of NAFLD.",
        "Русский": "Введите значения показателей для получения прогноза вероятности НАЖБП.",
    },
    "gender": {"English": "**Select Gender:**", "Русский": "**Выберите пол:**"},
    "male": {"English": "Male", "Русский": "Мужской"},
    "female": {"English": "Female", "Русский": "Женский"},
    "height": {"English": "**Height (cm):**", "Русский": "**Рост (см):**"},
    "weight": {"English": "**Weight (kg):**", "Русский": "**Вес (кг):**"},
    "bmi_calc": {"English": "BMI is calculated automatically.", "Русский": "ИМТ рассчитывается автоматически."},
    "calculate": {"English": "Calculate Prediction", "Русский": "Рассчитать Прогноз"},
    "probability": {"English": "Probability", "Русский": "Вероятность"},
    "class": {"English": "Class", "Русский": "Класс"},
    "healthy": {"English": "Healthy", "Русский": "Здоров"},
    "sick": {"English": "Sick", "Русский": "Болен"},
    "alert": {"English": "🚨 Model predicts that you are sick.", "Русский": "🚨 Модель предсказывает, что вы больны."},
    "success": {"English": "✅ Model predicts that you are healthy.", "Русский": "✅ Модель предсказывает, что вы здоровы."},
    "prediction results": {"English": "Prediction Results", "Русский": "Результаты Прогноза "}
}

# Extract expected feature names from the model
feature_names = ebm.term_names_

# Feature Translations
feature_translations = {
    "ИМТ": {"English": "BMI", "Русский": "ИМТ"},  # ✅ Added this missing key
    "Возраст": {"English": "Age", "Русский": "Возраст"},
    "О.ж.,%": {"English": "Total Fat %", "Русский": "Общий Жир,%"},
    "Висц.ж,%": {"English": "Visceral Fat %", "Русский": "Висц. Жир,%"},
    "Скелет,%": {"English": "Skeletal %", "Русский": "Скелет %"},
    "Кости,кг": {"English": "Bone Mass (kg)", "Русский": "Кости,кг"},
    "Вода,%": {"English": "Water %", "Русский": "Вода,%"},
    "СООВ,ккал": {"English": "Metabolic Rate (kcal)", "Русский": "Скорость Обмена Веществ,ккал"},
    "ОГ,см": {"English": "Chest Circumference", "Русский": "Обхват Груди,см"},
    "ОТ,см": {"English": "Waist Circumference", "Русский": "Обхват Талии,см"},
    "ОЖ,см": {"English": "Hip Circumference", "Русский": "Обхват Живота,см"},
    "ОБ,см": {"English": "Thigh Circumference", "Русский": "Обхват Бедра,см"},
    "АЛТ": {"English": "ALT", "Русский": "АЛТ"},
    "АСТ": {"English": "AST", "Русский": "АСТ"},
    "ГГТП": {"English": "GGT", "Русский": "ГГТП"},
    "ЩФ": {"English": "ALP", "Русский": "ЩФ"},
    "ХСобщ.": {"English": "Total Cholesterol", "Русский": "Холестерин Общ."},
    "ЛПНП": {"English": "LDL", "Русский": "ЛПНП"},
    "ЛПВП": {"English": "HDL", "Русский": "ЛПВП"},
    "Триглиц.": {"English": "Triglycerides", "Русский": "Триглиц."},
    "Билир.о": {"English": "Bilirubin (Total)", "Русский": "Билир. Общ."},
    "Билир.пр": {"English": "Bilirubin (Direct)", "Русский": "Билир. Прямой"},
    "Глюкоза": {"English": "Glucose", "Русский": "Глюкоза"},
    "Инсулин": {"English": "Insulin", "Русский": "Инсулин"},
    "Ферритин": {"English": "Ferritin", "Русский": "Ферритин"},
    "СРБ": {"English": "CRP", "Русский": "СРБ"},
    "О.белок": {"English": "Total Protein", "Русский": "Общий Белок"},
    "Моч.к-та": {"English": "Uric Acid", "Русский": "Моч.К-та"},
}

# Normal ranges for reference
normal_ranges = {
    'Пол': (0, 1),  # Binary: 0 (Мужской), 1 (Женский)
    'Возраст': (0, 100),
    'О.ж.,%': (10, 25),
    'Висц.ж,%': (5, 15),
    'Скелет,%': (30, 40),
    'Кости,кг': (2, 5),
    'Вода,%': (50, 70),
    'СООВ,ккал': (1500, 3500),
    'ОГ,см': (75, 105),
    'ОТ,см': (60, 95),
    'ОЖ,см': (70, 105),
    'ОБ,см': (40, 65),
    'ИМТ': (18.5, 24.9),
    'АЛТ': (7, 41),
    'АСТ': (10, 40),
    'ГГТП': (10, 70),
    'ЩФ': (40, 130),
    'ХСобщ.': (0, 5.2),
    'ЛПНП': (0, 3.0),
    'ЛПВП': (0.7, 2.3),
    'Триглиц.': (0, 1.7),
    'Билир.о': (0, 1.2),
    'Билир.пр': (0, 0.3),
    'Глюкоза': (3.9, 5.5),
    'Инсулин': (2.0, 26.0),  # Missing before
    'Ферритин': (10.0, 250.0),  # Missing before
    'СРБ': (0.0, 5.0),  # Missing before
    'О.белок': (55.0, 81.0),  # Missing before
    'Моч.к-та': (120.0, 420.0)  # Missing before
}

# Streamlit UI
st.title(translations["title"][lang])
st.write(translations["desc"][lang])

# Gender Selection
gender = st.radio(translations["gender"][lang], (translations["male"][lang], translations["female"][lang]))
gender_value = 0 if gender == translations["male"][lang] else 1

# Height & Weight Inputs
height = st.number_input(translations["height"][lang], min_value=100, max_value=250, value=170)
weight = st.number_input(translations["weight"][lang], min_value=30, max_value=200, value=70)
bmi = weight / ((height / 100) ** 2)  # BMI Calculation
st.write(f"**ИМТ / BMI:** {bmi:.2f} ({translations['bmi_calc'][lang]})")

user_input_dict = {
    'Пол': gender_value,
    'Возраст': st.number_input(f"**{feature_translations['Возраст'][lang]}**", min_value=0, max_value=100, value=50),
    'О.ж.,%': st.number_input(f"**{feature_translations['О.ж.,%'][lang]}**", min_value=0.0, max_value=70.0, value=20.0),
    'Висц.ж,%': st.number_input(f"**{feature_translations['Висц.ж,%'][lang]}**", min_value=0.0, max_value=50.0, value=10.0),
    'Скелет,%': st.number_input(f"**{feature_translations['Скелет,%'][lang]}**", min_value=0.0, max_value=100.0, value=35.0),
    'Кости,кг': st.number_input(f"**{feature_translations['Кости,кг'][lang]}**", min_value=0.0, max_value=20.0, value=3.0),
    'Вода,%': st.number_input(f"**{feature_translations['Вода,%'][lang]}**", min_value=0.0, max_value=100.0, value=60.0),
    'СООВ,ккал': st.number_input(f"**{feature_translations['СООВ,ккал'][lang]}**", min_value=0.0, max_value=7000.0, value=2000.0),
    'ОГ,см': st.number_input(f"**{feature_translations['ОГ,см'][lang]}**", min_value=0.0, max_value=150.0, value=90.0),
    'ОТ,см': st.number_input(f"**{feature_translations['ОТ,см'][lang]}**", min_value=0.0, max_value=150.0, value=80.0),
    'ОЖ,см': st.number_input(f"**{feature_translations['ОЖ,см'][lang]}**", min_value=0.0, max_value=150.0, value=90.0),
    'ОБ,см': st.number_input(f"**{feature_translations['ОБ,см'][lang]}**", min_value=0.0, max_value=150.0, value=50.0),
    'ИМТ': bmi,  # BMI is auto-calculated
    'АЛТ': st.number_input(f"**{feature_translations['АЛТ'][lang]}**", min_value=0.0, max_value=200.0, value=20.0),
    'АСТ': st.number_input(f"**{feature_translations['АСТ'][lang]}**", min_value=0.0, max_value=200.0, value=20.0),
    'ГГТП': st.number_input(f"**{feature_translations['ГГТП'][lang]}**", min_value=0.0, max_value=200.0, value=50.0),
    'ЩФ': st.number_input(f"**{feature_translations['ЩФ'][lang]}**", min_value=0.0, max_value=500.0, value=80.0),
    'ХСобщ.': st.number_input(f"**{feature_translations['ХСобщ.'][lang]}**", min_value=0.0, max_value=30.0, value=3.0),
    'ЛПНП': st.number_input(f"**{feature_translations['ЛПНП'][lang]}**", min_value=0.0, max_value=20.0, value=2.0),
    'ЛПВП': st.number_input(f"**{feature_translations['ЛПВП'][lang]}**", min_value=0.0, max_value=20.0, value=1.0),
    'Триглиц.': st.number_input(f"**{feature_translations['Триглиц.'][lang]}**", min_value=0.0, max_value=50.0, value=1.5),
    'Билир.о': st.number_input(f"**{feature_translations['Билир.о'][lang]}**", min_value=0.0, max_value=30.0, value=1.0),
    'Билир.пр': st.number_input(f"**{feature_translations['Билир.пр'][lang]}**", min_value=0.0, max_value=30.0, value=0.5),
    'Глюкоза': st.number_input(f"**{feature_translations['Глюкоза'][lang]}**", min_value=0.0, max_value=50.0, value=5.0),
    'Инсулин': st.number_input(f"**{feature_translations['Инсулин'][lang]}**", min_value=0.0, max_value=100.0, value=5.0),
    'Ферритин': st.number_input(f"**{feature_translations['Ферритин'][lang]}**", min_value=0.0, max_value=1000.0, value=150.0),
    'СРБ': st.number_input(f"**{feature_translations['СРБ'][lang]}**", min_value=0.0, max_value=20.0, value=1.0),
    'О.белок': st.number_input(f"**{feature_translations['О.белок'][lang]}**", min_value=0.0, max_value=150.0, value=70.0),
    'Моч.к-та': st.number_input(f"**{feature_translations['Моч.к-та'][lang]}**", min_value=0.0, max_value=600.0, value=200.0)
}
user_input_dict['ИМТ'] = bmi  # BMI is auto-calculated
scaled_input_dict = {feature: min_max_scaler(value, feature) for feature, value in user_input_dict.items()}

try:
    input_df = pd.DataFrame([scaled_input_dict])
    input_df = input_df[ebm.feature_names_in_]
    input_array = input_df.to_numpy()
except KeyError as e:
    st.error(f"Missing required features: {e}")
    st.stop()

# Check if all features exist in the model
model_features = set(ebm.feature_names_in_)
csv_features = set(feature_min.keys())

missing_features = model_features - csv_features
extra_features = csv_features - model_features

if missing_features:
    st.warning(f"⚠️ Missing features in CSV: {missing_features}")
# if extra_features:
#     st.warning(f"⚠️ Extra features in CSV that model doesn't expect: {extra_features}")

# Predict probability and classify
if st.button(translations["calculate"][lang]):
    try:
        probability = ebm.predict_proba(input_array)[0][1]
        predicted_class = translations["sick"][lang] if probability >= 0.5 else translations["healthy"][lang]
        st.success(f"{translations['probability'][lang]}: {probability:.4f} ({predicted_class})")
    except Exception as e:
        st.error(f"Ошибка / Error: {e}")

    st.subheader(translations["prediction results"][lang])
    st.write(f"**{translations['probability'][lang]}:** {probability:.4f}")
    st.write(f"**{translations['class'][lang]}:** {predicted_class}")

    if predicted_class == translations["sick"][lang]:
        st.error(translations["alert"][lang])
    else:
        st.success(translations["success"][lang])

    # Normalize user values for comparison (excluding Gender and Age)
    plot_features = [key for key in normal_ranges.keys() if key not in ["Пол", "Возраст"]]  # Exclude "Пол" (Gender) & "Возраст" (Age)
    normal_min = [normal_ranges[key][0] for key in plot_features]
    normal_max = [normal_ranges[key][1] for key in plot_features]
    user_values = [user_input_dict[key] for key in plot_features]  # Get user-entered values

    # Normalize function
    def normalize(values, min_vals, max_vals):
        return [(val - min_val) / (max_val - min_val) if max_val != min_val else 0 for val, min_val, max_val in zip(values, min_vals, max_vals)]

    normalized_user_values = normalize(user_values, normal_min, normal_max)

    # Plot comparison graph
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot normal range bars
    normal_range_added = False  

    for i, (min_val, max_val) in enumerate(zip([0] * len(normal_min), [1] * len(normal_max))):
        ax.barh(
            i, 
            max_val - min_val, 
            left=min_val, 
            color='gray', 
            alpha=0.5, 
            label=("Normal Range" if lang == "English" else "Норма") if not normal_range_added else "", 
            height=0.5
        )
        normal_range_added = True  # Ensure the label is added only once
    # Plot user values
    for i, value in enumerate(normalized_user_values):
        ax.scatter(value, i, color='blue', s=100, zorder=5)

    # Add only ONE legend entry for 'Your Value' / 'Ваше значение'
    ax.scatter([], [], color='blue', s=100, label="Your Value" if lang == "English" else "Ваше значение")

    # Translate feature labels
    translated_labels = [feature_translations[feat][lang] for feat in plot_features]

    # Formatting
    ax.set_xlim([-0.1, 1.1])
    ax.get_xaxis().set_visible(False)
    ax.set_xlabel(translations["probability"][lang], fontsize=12, fontweight='bold')
    ax.set_title(translations["title"][lang], fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(plot_features)))
    ax.set_yticklabels(translated_labels, fontsize=11, fontweight='bold')

    ax.legend(loc='upper left', fontsize=10)
    st.pyplot(fig)


