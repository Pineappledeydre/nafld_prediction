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

# Extract expected feature names from the model
feature_names = ebm.term_names_

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

user_input_dict = {
    'Пол': gender_value,
    'Возраст': st.number_input("**Возраст**", min_value=0, max_value=100, value=30),
    'О.ж.,%': st.number_input("**О.ж.,%**", min_value=0.0, max_value=100.0, value=20.0),
    'Висц.ж,%': st.number_input("**Висц.ж,%**", min_value=0.0, max_value=100.0, value=5.0),
    'Скелет,%': st.number_input("**Скелет,%**", min_value=0.0, max_value=100.0, value=40.0),
    'Кости,кг': st.number_input("**Кости,кг**", min_value=0.0, max_value=20.0, value=3.0),
    'Вода,%': st.number_input("**Вода,%**", min_value=0.0, max_value=100.0, value=60.0),
    'СООВ,ккал': st.number_input("**СООВ,ккал**", min_value=0.0, max_value=5000.0, value=2000.0),
    'ОГ,см': st.number_input("**ОГ,см**", min_value=0.0, max_value=150.0, value=90.0),
    'ОТ,см': st.number_input("**ОТ,см**", min_value=0.0, max_value=150.0, value=80.0),
    'ОЖ,см': st.number_input("**ОЖ,см**", min_value=0.0, max_value=150.0, value=100.0),
    'ОБ,см': st.number_input("**ОБ,см**", min_value=0.0, max_value=150.0, value=55.0),
    'ИМТ': st.number_input("**ИМТ**", min_value=0.0, max_value=100.0, value=24.0),
    'АЛТ': st.number_input("**АЛТ**", min_value=0.0, max_value=200.0, value=30.0),
    'АСТ': st.number_input("**АСТ**", min_value=0.0, max_value=200.0, value=30.0),
    'ГГТП': st.number_input("**ГГТП**", min_value=0.0, max_value=200.0, value=15.0),
    'ЩФ': st.number_input("**ЩФ**", min_value=0.0, max_value=500.0, value=100.0),
    'ХСобщ.': st.number_input("**ХСобщ.**", min_value=0.0, max_value=500.0, value=200.0),
    'ЛПНП': st.number_input("**ЛПНП**", min_value=0.0, max_value=100.0, value=3.0),
    'ЛПВП': st.number_input("**ЛПВП**", min_value=0.0, max_value=100.0, value=1.0),
    'Триглиц.': st.number_input("**Триглиц.**", min_value=0.0, max_value=100.0, value=1.5),
    'Билир.о': st.number_input("**Билир.о**", min_value=0.0, max_value=10.0, value=1.0),
    'Билир.пр': st.number_input("**Билир.пр**", min_value=0.0, max_value=100.0, value=0.5),
    'Глюкоза': st.number_input("**Глюкоза**", min_value=0.0, max_value=100.0, value=5.0),
    'Инсулин': st.number_input("**Инсулин**", min_value=0.0, max_value=100.0, value=5.0),
    'Ферритин': st.number_input("**Ферритин**", min_value=0.0, max_value=1000.0, value=50.0),
    'СРБ': st.number_input("**СРБ**", min_value=0.0, max_value=10.0, value=1.0),
    'О.белок': st.number_input("**О.белок**", min_value=0.0, max_value=10.0, value=7.0),
    'Моч.к-та': st.number_input("**Моч.к-та**", min_value=0.0, max_value=100.0, value=5.0)
}

# Compute interaction terms
user_input_dict.update({
    'Возраст & ОГ,см': user_input_dict['Возраст'] * user_input_dict['ОГ,см'],
    'Возраст & ОТ,см': user_input_dict['Возраст'] * user_input_dict['ОТ,см'],
    'Возраст & ОЖ,см': user_input_dict['Возраст'] * user_input_dict['ОЖ,см'],
    'Возраст & АЛТ': user_input_dict['Возраст'] * user_input_dict['АЛТ'],
    'Возраст & ГГТП': user_input_dict['Возраст'] * user_input_dict['ГГТП'],
    'Возраст & Ферритин': user_input_dict['Возраст'] * user_input_dict['Ферритин'],
    'О.ж.,% & АЛТ': user_input_dict['О.ж.,%'] * user_input_dict['АЛТ'],
    'О.ж.,% & Билир.о': user_input_dict['О.ж.,%'] * user_input_dict['Билир.о'],
    'О.ж.,% & Глюкоза': user_input_dict['О.ж.,%'] * user_input_dict['Глюкоза'],
    'Висц.ж,% & АЛТ': user_input_dict['Висц.ж,%'] * user_input_dict['АЛТ'],
    'Висц.ж,% & СРБ': user_input_dict['Висц.ж,%'] * user_input_dict['СРБ'],
    'Скелет,% & АЛТ': user_input_dict['Скелет,%'] * user_input_dict['АЛТ'],
    'Скелет,% & ГГТП': user_input_dict['Скелет,%'] * user_input_dict['ГГТП'],
    'ОГ,см & СРБ': user_input_dict['ОГ,см'] * user_input_dict['СРБ'],
    'ОТ,см & ЩФ': user_input_dict['ОТ,см'] * user_input_dict['ЩФ'],
    'ОТ,см & СРБ': user_input_dict['ОТ,см'] * user_input_dict['СРБ'],
    'ОБ,см & ИМТ': user_input_dict['ОБ,см'] * user_input_dict['ИМТ'],
    'ИМТ & АЛТ': user_input_dict['ИМТ'] * user_input_dict['АЛТ'],
    'ИМТ & ГГТП': user_input_dict['ИМТ'] * user_input_dict['ГГТП'],
    'ИМТ & ЛПВП': user_input_dict['ИМТ'] * user_input_dict['ЛПВП'],
    'ИМТ & Инсулин': user_input_dict['ИМТ'] * user_input_dict['Инсулин'],
    'АЛТ & Билир.о': user_input_dict['АЛТ'] * user_input_dict['Билир.о'],
    'АЛТ & О.белок': user_input_dict['АЛТ'] * user_input_dict['О.белок'],
    'АСТ & ГГТП': user_input_dict['АСТ'] * user_input_dict['ГГТП'],
    'ГГТП & Билир.о': user_input_dict['ГГТП'] * user_input_dict['Билир.о'],
    'ГГТП & Билир.пр': user_input_dict['ГГТП'] * user_input_dict['Билир.пр'],
    'ЛПНП & Билир.пр': user_input_dict['ЛПНП'] * user_input_dict['Билир.пр']
})

# Convert input dictionary to DataFrame
input_df = pd.DataFrame([user_input_dict])

# Ensure the column order matches the model's expected features
input_df = input_df[ebm.term_names_]

# Convert to NumPy array
input_array = input_df.to_numpy()


# Predict probability and classify
if st.button("Рассчитать Прогноз"):
    try:
        probability = ebm.predict_proba(input_array)[0][1]
        predicted_class = "Болен" if probability >= 0.5 else "Здоров"
        st.success(f"Вероятность: {probability:.4f} ({predicted_class})")
    except Exception as e:
        st.error(f"Ошибка: {e}")

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

    normalized_user_values = normalize(input_values[1:], normal_min, normal_max)

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
