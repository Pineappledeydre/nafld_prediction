import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

unscaled_coefficients = {
    'Возраст': -0.052959,
    'Висц.ж,%': 0.408395,
    'Мыш.м,%': -0.169631,
    'ИМТ': 0.178240,
    'АЛТ': 0.119727,
    'АСТ': 0.060623,
    'ГГТП': 0.090496,
    'ЛПНП': -0.550825,
    'Ферритин': -0.005117,
    'О.белок': -0.107443
}

unscaled_intercept = 5.67207853

def predict_probability_and_class(features, threshold=0.5):
    y_pred = unscaled_intercept
    for feature, coef in unscaled_coefficients.items():
        y_pred += coef * features.get(feature, 0)
    probability = 1 / (1 + np.exp(-y_pred))
    predicted_class = "Болен" if probability >= threshold else "Здоров"
    return probability, predicted_class

st.set_page_config(page_title="Прогноз НАЖБП", page_icon="💉", layout="wide")

st.title("💉 Прогноз НАЖБП")
st.write("Введите значения показателей для получения прогноза вероятности НАЖБП.")

gender = st.radio("**Выберите пол:**", ("Мужской", "Женский"))

if gender == "Мужской":
    normal_ranges = {
        'Возраст': (0, 100),
        'Висцеральный жир, %': (1, 10),
        'Мышечная масса, %': (40, 50),
        'ИМТ': (18.5, 24.9),
        'АЛТ (ед/л)': (7, 42),
        'АСТ (ед/л)': (7, 42),
        'ГГТП (ед/л)': (9, 40),
        'ЛПНП (ммоль/л)': (2, 5),
        'Ферритин (нг/мл)': (20, 250),
        'Общий белок (г/л)': (60, 85)
    }
else:
    normal_ranges = {
        'Возраст': (0, 100),
        'Висцеральный жир, %': (1, 10),
        'Мышечная масса, %': (30, 40),
        'ИМТ': (18.5, 24.9),
        'АЛТ (ед/л)': (7, 42),
        'АСТ (ед/л)': (7, 42),
        'ГГТП (ед/л)': (9, 32),
        'ЛПНП (ммоль/л)': (1.9, 4.6),
        'Ферритин (нг/мл)': (10, 130),
        'Общий белок (г/л)': (60, 85)
    }

col1, col2 = st.columns(2)

with col1:
    st.header("**Введите показатели:**")
    age = st.number_input("**Возраст**", min_value=0, max_value=100, value=30)
    vis_fat = st.number_input("**Висцеральный жир, %**", min_value=0.0, max_value=100.0, value=5.0)
    muscle_mass = st.number_input("**Мышечная масса, %**", min_value=0.0, max_value=100.0, value=30.0)
    bmi = st.number_input("**Индекс массы тела (ИМТ)**", min_value=0.0, max_value=100.0, value=24.0)
    alt = st.number_input("**АЛТ (ед/л)**", min_value=0.0, max_value=200.0, value=30.0)
    ast = st.number_input("**АСТ (ед/л)**", min_value=0.0, max_value=200.0, value=30.0)
    ggtp = st.number_input("**ГГТП (ед/л)**", min_value=0.0, max_value=200.0, value=15.0)
    ldl = st.number_input("**ЛПНП (ммоль/л)**", min_value=0.0, max_value=10.0, value=3.0)
    ferritin = st.number_input("**Ферритин (нг/мл)**", min_value=0.0, max_value=1000.0, value=50.0)
    total_protein = st.number_input("**Общий белок (г/л)**", min_value=0.0, max_value=100.0, value=70.0)

with col2:
    if st.button("Рассчитать Прогноз"):
        input_features = {
            'Возраст': age,
            'Висц.ж,%': vis_fat,
            'Мыш.м,%': muscle_mass,
            'ИМТ': bmi,
            'АЛТ': alt,
            'АСТ': ast,
            'ГГТП': ggtp,
            'ЛПНП': ldl,
            'Ферритин': ferritin,
            'О.белок': total_protein
        }
        probability, predicted_class = predict_probability_and_class(input_features)

        st.subheader("**Результаты Прогноза:**")
        st.write(f"**Вероятность:** {probability:.4f}")
        st.write(f"**Класс:** {predicted_class}")

        if predicted_class == "Болен":
            st.error("Модель предсказывает, что вы больны.")
        else:
            st.success("Модель предсказывает, что вы здоровы.")

        st.subheader("**Сравнение введенных значений с нормальными диапазонами**")
        fig, ax = plt.subplots(figsize=(8, 5))
        features = ['Возраст', 'Висцеральный жир, %', 'Мышечная масса, %', 'ИМТ', 'АЛТ (ед/л)', 'АСТ (ед/л)', 'ГГТП (ед/л)', 'ЛПНП', 'Ферритин', 'Общий белок (г/л)']
        user_values = [age, vis_fat, muscle_mass, bmi, alt, ast, ggtp, ldl, ferritin, total_protein]
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
