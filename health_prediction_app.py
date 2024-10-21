import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json

intercept = 4.5729182157694765

with open('contributions.json', 'r') as f:
    contributions = json.load(f)
contributions = [np.array(c) for c in contributions]  

with open('bin_edges.json', 'r') as f:
    bin_edges = json.load(f)

feature_min = np.array([0.000e+00, 2.500e+01, 1.770e+01, 5.000e+00, 2.710e+01, 2.000e+00,
                        3.070e+01, 1.179e+03, 8.900e+01, 7.600e+01, 8.400e+01, 8.600e+01,
                        2.093e+01, 7.100e+00, 9.000e+00, 8.000e+00, 4.200e+01, 3.260e+00,
                        1.300e+00, 7.000e-01, 3.600e-01, 4.400e+00, 1.900e+00, 4.100e+00])

feature_max = np.array([1.000e+00, 6.000e+01, 4.740e+01, 2.600e+01, 4.870e+01, 4.700e+00,
                        6.140e+01, 2.156e+03, 1.320e+02, 1.340e+02, 1.350e+02, 1.340e+02,
                        4.232e+01, 1.237e+02, 1.040e+02, 1.440e+02, 1.560e+02, 7.800e+00,
                        5.900e+00, 2.800e+00, 5.510e+00, 3.140e+01, 1.740e+01, 7.900e+00])

def predict_log_odds(intercept, contributions, feature_values, bin_edges):
    log_odds = intercept
    for i, value in enumerate(feature_values):
        bin_idx = np.digitize(value, bin_edges[i]) - 1
        bin_idx = min(bin_idx, len(contributions[i]) - 1)
        log_odds += contributions[i][bin_idx]
    return log_odds

def predict_probability(log_odds):
    return 1 / (1 + np.exp(-log_odds))

def min_max_scaler(values, feature_min, feature_max):
    scaled_values = (values - feature_min) / (feature_max - feature_min)
    return scaled_values

# Streamlit page setup
st.set_page_config(page_title="Прогноз НАЖБП", page_icon="💉", layout="wide")

st.title("💉 Прогноз НАЖБП")
st.write("Введите значения показателей для получения прогноза вероятности НАЖБП.")

gender = st.radio("**Выберите пол:**", ("Мужской", "Женский"))
gender_value = 0 if gender == "Мужской" else 1

if gender == "Мужской":
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
else:
    normal_ranges = {
        'Возраст': (0, 100),
        'О.ж.,%': (18, 30),
        'Висц.ж,%': (5, 15),
        'Скелет,%': (25, 35),
        'Кости,кг': (1.5, 4),
        'Вода,%': (50, 60),
        'СООВ,ккал': (1200, 3000),
        'ОГ,см': (75, 100),
        'ОТ,см': (60, 90),
        'ОЖ,см': (70, 100),
        'ОБ,см': (35, 60),
        'ИМТ': (18.5, 24.9),
        'АЛТ': (7, 35),
        'АСТ': (10, 35),
        'ГГТП': (10, 50),
        'ЩФ': (40, 120),
        'ХСобщ.': (0, 5.2),
        'ЛПНП': (0, 3.0),
        'ЛПВП': (1.0, 1.5),
        'Триглиц.': (0, 1.7),
        'Билир.о': (0, 1.2),
        'Билир.пр': (0, 0.3),
        'Глюкоза': (3.9, 5.5)
    }

col1, col2 = st.columns(2)

with col1:
    st.header("**Введите показатели:**")
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

with col2:
    alkaline_phosphatase = st.number_input("**ЩФ**", min_value=0.0, max_value=500.0, value=100.0)
    cholesterol = st.number_input("**ХСобщ.**", min_value=0.0, max_value=500.0, value=200.0)
    ldl = st.number_input("**ЛПНП**", min_value=0.0, max_value=100.0, value=3.0)
    hdl = st.number_input("**ЛПВП**", min_value=0.0, max_value=100.0, value=1.0)
    triglycerides = st.number_input("**Триглиц.**", min_value=0.0, max_value=100.0, value=1.5)
    bilirubin_total = st.number_input("**Билир.о**", min_value=0.0, max_value=10.0, value=1.0)
    bilirubin_direct = st.number_input("**Билир.пр**", min_value=0.0, max_value=100.0, value=0.5)
    glucose = st.number_input("**Глюкоза**", min_value=0.0, max_value=100.0, value=5.0)

    if st.button("Рассчитать Прогноз"):
        input_features = [
            gender_value, age, total_fat, vis_fat, skeletal_mass, bone_mass, 
            water, metabolic_rate, chest, waist, hip, thigh, bmi, alt, ast, 
            ggtp, alkaline_phosphatase, cholesterol, ldl, hdl, triglycerides, 
            bilirubin_total, bilirubin_direct, glucose
        ]

        scaled_features = min_max_scaler(np.array(input_features), np.array(feature_min), np.array(feature_max))
        log_odds = predict_log_odds(intercept, contributions, scaled_features, bin_edges)
        probability = predict_probability(log_odds)

        predicted_class = "Болен" if probability >= 0.5 else "Здоров"

        st.subheader("**Результаты Прогноза:**")
        st.write(f"**Вероятность:** {probability:.4f}")
        st.write(f"**Класс:** {predicted_class}")

        if predicted_class == "Болен":
            st.error("Модель предсказывает, что вы больны.")
        else:
            st.success("Модель предсказывает, что вы здоровы.")

        st.subheader("**Сравнение введенных значений с нормальными диапазонами**")
        fig, ax = plt.subplots(figsize=(8, 5))
        features = ['Возраст', 'О.ж.,%', 'Висц.ж,%', 'Скелет,%', 'Кости,кг', 'Вода,%', 'СООВ,ккал', 'ОГ,см', 
                    'ОТ,см', 'ОЖ,см', 'ОБ,см', 'ИМТ', 'АЛТ', 'АСТ', 'ГГТП', 'ЩФ', 'ХСобщ.', 'ЛПНП', 'ЛПВП', 
                    'Триглиц.', 'Билир.о', 'Билир.пр', 'Глюкоза']
        user_values = input_features[1:]
        normal_min = [normal_ranges[feat][0] for feat in features]
        normal_max = [normal_ranges[feat][1] for feat in features]

        for i, (min_val, max_val) in enumerate(zip(normal_min, normal_max)):
            ax.plot([min_val, max_val], [i, i], color='gray', lw=6, alpha=0.5, label='Нормальный диапазон' if i == 0 else "")

        ax.scatter(user_values, range(len(features)), color='blue', s=100, zorder=5, label='Ваши значения')

        ax.set_xlabel('Значения', fontsize=12, fontweight='bold')
        ax.set_title('Сравнение показателей с нормальными диапазонами', fontsize=14, fontweight='bold')

        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', labelsize=10)
        ax.xaxis.label.set_size(12)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        ax.grid(True, axis='x', linestyle='--', alpha=0.6)
        ax.legend(loc='lower right', fontsize=10)

        st.pyplot(fig)

st.write("---")
st.write("Это приложение создано для помощи в прогнозе НАЖБП на основе различных показателей.")
st.write("Приложение использует биохимические маркеры, индекс массы тела, и другие параметры.")
st.write("Приложение НЕ создано для самостоятельной постановки диагноза.")
