import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier

# Load the trained EBM model
MODEL_PATH = "models/ebm_model.pkl"

with open(MODEL_PATH, "rb") as file:
    ebm = pickle.load(file)

# Language Selection
lang = st.radio("🌍 **Select Language / Выберите язык:**", ("English", "Русский"))

# Translations
translations = {
    "title": {"English": "💉 NAFLD Prediction", "Русский": "💉 Прогноз НАЖБП"},
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
}

# Extract expected feature names from the model
feature_names = ebm.term_names_

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
    'ЛПВП': (1.0, 1.5),
    'Триглиц.': (0, 1.7),
    'Билир.о': (0, 1.2),
    'Билир.пр': (0, 0.3),
    'Глюкоза': (3.9, 5.5),
    'Инсулин': (2.0, 25.0),  # Missing before
    'Ферритин': (15, 300),  # Missing before
    'СРБ': (0, 3),  # Missing before
    'О.белок': (6.0, 8.5),  # Missing before
    'Моч.к-та': (2.4, 7.0)  # Missing before
}

# Streamlit UI
st.set_page_config(page_title=translations["title"][lang], page_icon="💉", layout="wide")
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
    'Возраст': st.number_input("**Возраст**", min_value=0, max_value=100, value=50),
    'О.ж.,%': st.number_input("**О.ж.,%**", min_value=0.0, max_value=70.0, value=20.0),
    'Висц.ж,%': st.number_input("**Висц.ж,%**", min_value=0.0, max_value=50.0, value=10.0),
    'Скелет,%': st.number_input("**Скелет,%**", min_value=0.0, max_value=100.0, value=35.0),
    'Кости,кг': st.number_input("**Кости,кг**", min_value=0.0, max_value=20.0, value=3.0),
    'Вода,%': st.number_input("**Вода,%**", min_value=0.0, max_value=100.0, value=60.0),
    'СООВ,ккал': st.number_input("**СООВ,ккал**", min_value=0.0, max_value=7000.0, value=2000.0),
    'ОГ,см': st.number_input("**ОГ,см**", min_value=0.0, max_value=150.0, value=90.0),
    'ОТ,см': st.number_input("**ОТ,см**", min_value=0.0, max_value=150.0, value=80.0),
    'ОЖ,см': st.number_input("**ОЖ,см**", min_value=0.0, max_value=150.0, value=90.0),
    'ОБ,см': st.number_input("**ОБ,см**", min_value=0.0, max_value=150.0, value=50.0),
    'ИМТ': bmi,  # BMI is auto-calculated
    'АЛТ': st.number_input("**АЛТ**", min_value=0.0, max_value=200.0, value=20.0),
    'АСТ': st.number_input("**АСТ**", min_value=0.0, max_value=200.0, value=20.0),
    'ГГТП': st.number_input("**ГГТП**", min_value=0.0, max_value=200.0, value=50.0),
    'ЩФ': st.number_input("**ЩФ**", min_value=0.0, max_value=500.0, value=80.0),
    'ХСобщ.': st.number_input("**ХСобщ.**", min_value=0.0, max_value=30.0, value=3.0),
    'ЛПНП': st.number_input("**ЛПНП**", min_value=0.0, max_value=20.0, value=2.0),
    'ЛПВП': st.number_input("**ЛПВП**", min_value=0.0, max_value=20.0, value=2.0),
    'Триглиц.': st.number_input("**Триглиц.**", min_value=0.0, max_value=50.0, value=1.5),
    'Билир.о': st.number_input("**Билир.о**", min_value=0.0, max_value=30.0, value=1.0),
    'Билир.пр': st.number_input("**Билир.пр**", min_value=0.0, max_value=30.0, value=0.5),
    'Глюкоза': st.number_input("**Глюкоза**", min_value=0.0, max_value=50.0, value=5.0),
    'Инсулин': st.number_input("**Инсулин**", min_value=0.0, max_value=100.0, value=5.0),
    'Ферритин': st.number_input("**Ферритин**", min_value=0.0, max_value=1000.0, value=150.0),
    'СРБ': st.number_input("**СРБ**", min_value=0.0, max_value=20.0, value=1.0),
    'О.белок': st.number_input("**О.белок**", min_value=0.0, max_value=20.0, value=7.0),
    'Моч.к-та': st.number_input("**Моч.к-та**", min_value=0.0, max_value=50.0, value=5.0)
}

# Convert input dictionary to DataFrame
input_df = pd.DataFrame([user_input_dict])
input_df = input_df[ebm.feature_names_in_]
# Convert to NumPy array
input_array = input_df.to_numpy()

# Debugging
print(f"✅ Model Expected Features: {ebm.feature_names_in_}")
print(f"✅ Input Data Features: {list(input_df.columns)}")
print(f"✅ Final input shape: {input_array.shape}")  # Must match (1, 29)

# Predict probability and classify
if st.button(translations["calculate"][lang]):
    try:
        probability = ebm.predict_proba(input_array)[0][1]
        predicted_class = translations["sick"][lang] if probability >= 0.5 else translations["healthy"][lang]
        st.success(f"{translations['probability'][lang]}: {probability:.4f} ({predicted_class})")
    except Exception as e:
        st.error(f"Ошибка / Error: {e}")

    st.subheader("**Результаты Прогноза / Prediction Results:**")
    st.write(f"**{translations['probability'][lang]}:** {probability:.4f}")
    st.write(f"**{translations['class'][lang]}:** {predicted_class}")

    if predicted_class == translations["sick"][lang]:
        st.error(translations["alert"][lang])
    else:
        st.success(translations["success"][lang])

    # Normalize user values for comparison
    feature_keys = list(normal_ranges.keys())
    normal_min = [normal_ranges[key][0] for key in feature_keys]
    normal_max = [normal_ranges[key][1] for key in feature_keys]

    def normalize(values, min_vals, max_vals):
        return [(val - min_val) / (max_val - min_val) for val, min_val, max_val in zip(values, min_vals, max_vals)]

    normalized_user_values = normalize(input_df.iloc[0].values, normal_min, normal_max)
    # Plot comparison graph
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, (min_val, max_val) in enumerate(zip([0] * len(normal_min), [1] * len(normal_max))):
        ax.barh(i, max_val - min_val, left=min_val, color='gray', alpha=0.5, label='Норма' if i == 0 else "", height=0.5)

    for i, value in enumerate(normalized_user_values):
        ax.scatter(value, i, color='blue', s=100, zorder=5, label='Ваше значение' if i == 0 else "")

    ax.set_xlim([-0.5, 1.5])
    ax.get_xaxis().set_visible(False)
    ax.set_xlabel('Нормализованные значения (0-1)', fontsize=12, fontweight='bold')
    ax.set_title('Сравнение показателей с нормальными диапазонами', fontsize=14, fontweight='bold')

    ax.set_yticks(range(len(feature_keys)))
    ax.set_yticklabels(feature_keys, fontsize=11, fontweight='bold')

    ax.legend(loc='upper left', fontsize=10)
    plt.show()

    st.pyplot(fig)

