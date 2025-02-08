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

feature_names = [
    'Пол', 'Возраст', 'О.ж.,%', 'Висц.ж,%', 'Скелет,%', 'Кости,кг', 'Вода,%', 'СООВ,ккал',
    'ОГ,см', 'ОТ,см', 'ОЖ,см', 'ОБ,см', 'ИМТ', 'АЛТ', 'АСТ', 'ГГТП', 'ЩФ', 'ХСобщ.', 'ЛПНП',
    'ЛПВП', 'Триглиц.', 'Билир.о', 'Билир.пр', 'Глюкоза'
]

# Normal ranges for reference
normal_ranges = {
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
    'Глюкоза': (3.9, 5.5)
}

# Streamlit UI
st.set_page_config(page_title="Прогноз НАЖБП", page_icon="💉", layout="wide")
st.title("💉 Прогноз НАЖБП")
st.write("Введите значения показателей для получения прогноза вероятности НАЖБП.")

# User Input Fields
gender = st.radio("**Выберите пол:**", ("Мужской", "Женский"))
gender_value = 0 if gender == "Мужской" else 1

age = st.number_input("**Возраст**", min_value=0, max_value=100, value=30)
total_fat = st.number_input("**О.ж.,%**", min_value=0.0, max_value=100.0, value=20.0)
vis_fat = st.number_input("**Висц.ж,%**", min_value=0.0, max_value=100.0, value=5.0)
skeletal_mass = st.number_input("**Скелет,%**", min_value=0.0, max_value=100.0, value=40.0)
bone_mass = st.number_input("**Кости,кг**", min_value=0.0, max_value=20.0, value=3.0)
water = st.number_input("**Вода,%**", min_value=0.0, max_value=100.0, value=60.0)
metabolic_rate = st.number_input("**СООВ,ккал**", min_value=0.0, max_value=5000.0, value=2000.0)
chest = st.number_input("**ОГ,см**", min_value=0.0, max_value=150.0, value=90.0)
waist = st.number_input("**ОТ,см**", min_value=0.0, max_value=150.0, value=80.0)
hip = st.number_input("**ОЖ,см**", min_value=0.0, max_value=150.0, value=100.0)
thigh = st.number_input("**ОБ,см**", min_value=0.0, max_value=150.0, value=55.0)
bmi = st.number_input("**ИМТ**", min_value=0.0, max_value=100.0, value=24.0)
alt = st.number_input("**АЛТ**", min_value=0.0, max_value=200.0, value=30.0)
ast = st.number_input("**АСТ**", min_value=0.0, max_value=200.0, value=30.0)
ggtp = st.number_input("**ГГТП**", min_value=0.0, max_value=200.0, value=15.0)

# Collect user input
user_values = [
    gender_value, age, total_fat, vis_fat, skeletal_mass, bone_mass, water, metabolic_rate, chest, waist, hip, thigh,
    bmi, alt, ast, ggtp
]

# Predict probability and classify
if st.button("Рассчитать Прогноз"):
    input_array = np.array(user_values).reshape(1, -1)
    probability = ebm.predict_proba(input_array)[0][1]
    predicted_class = "Болен" if probability >= 0.5 else "Здоров"

    st.subheader("**Результаты Прогноза:**")
    st.write(f"**Вероятность:** {probability:.4f}")
    st.write(f"**Класс:** {predicted_class}")

    if predicted_class == "Болен":
        st.error("🚨 Модель предсказывает, что вы больны.")
    else:
        st.success("✅ Модель предсказывает, что вы здоровы.")

    # Normalize user values for comparison
    feature_keys = list(normal_ranges.keys())
    normal_min = [normal_ranges[key][0] for key in feature_keys]
    normal_max = [normal_ranges[key][1] for key in feature_keys]

    def normalize(values, min_vals, max_vals):
        return [(val - min_val) / (max_val - min_val) for val, min_val, max_val in zip(values, min_vals, max_vals)]

    normalized_user_values = normalize(user_values[1:], normal_min, normal_max)

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
