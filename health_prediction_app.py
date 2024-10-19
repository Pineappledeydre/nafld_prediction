
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Define the unscaled coefficients and intercept
unscaled_coefficients = {
    'О.ж.,%': 0.011731,
    'Висц.ж,%': 0.256571,
    'Мыш.м,%': -0.118061,
    'ИМТ': 0.175329,
    'АЛТ': 0.109809,
    'АСТ': 0.060320,
    'ГГТП': 0.063493,
    'О.белок': -0.118442
}

unscaled_intercept = 1.01741283

# Function to predict probability and class based on input features
def predict_probability_and_class(features, threshold=0.5):
    y_pred = unscaled_intercept
    for feature, coef in unscaled_coefficients.items():
        y_pred += coef * features.get(feature, 0)
    probability = 1 / (1 + np.exp(-y_pred))
    predicted_class = "Болен" if probability >= threshold else "Здоров"
    return probability, predicted_class

# Streamlit App Interface
st.set_page_config(page_title="Прогноз НАЖБП", page_icon="💉", layout="wide")

# App Title and Description
st.title("💉 Прогноз НАЖБП")
st.write("Введите значения показателей для получения прогноза вероятности НАЖБП.")

# Gender selection
gender = st.radio("**Выберите пол:**", ("Мужской", "Женский"))

# Adjust normal ranges based on gender
if gender == "Мужской":
    normal_ranges = {
        'Ожирение, %': (10, 20),
        'Висцеральный жир, %': (5, 15),
        'Мышечная масса, %': (40, 50),
        'ИМТ': (18.5, 24.9),
        'АЛТ (ед/л)': (7, 56),
        'АСТ (ед/л)': (8, 48),
        'ГГТП (ед/л)': (9, 48),
        'Общий белок (г/л)': (6.0, 8.3)
    }
else:
    normal_ranges = {
        'Ожирение, %': (18, 28),
        'Висцеральный жир, %': (5, 15),
        'Мышечная масса, %': (30, 40),
        'ИМТ': (18.5, 24.9),
        'АЛТ (ед/л)': (7, 56),
        'АСТ (ед/л)': (8, 48),
        'ГГТП (ед/л)': (9, 48),
        'Общий белок (г/л)': (6.0, 8.3)
    }

# Layout for input and results
col1, col2 = st.columns(2)

with col1:
    st.header("**Введите показатели:**")
    О_ж = st.number_input("**Ожирение, %**", min_value=0.0, max_value=100.0, value=25.0)
    Висц_ж = st.number_input("**Висцеральный жир, %**", min_value=0.0, max_value=100.0, value=15.0)
    Мыш_м = st.number_input("**Мышечная масса, %**", min_value=0.0, max_value=100.0, value=45.0)
    ИМТ = st.number_input("**Индекс массы тела (ИМТ)**", min_value=0.0, max_value=100.0, value=24.0)
    АЛТ = st.number_input("**АЛТ (ед/л)**", min_value=0.0, max_value=200.0, value=30.0)
    АСТ = st.number_input("**АСТ (ед/л)**", min_value=0.0, max_value=200.0, value=20.0)
    ГГТП = st.number_input("**ГГТП (ед/л)**", min_value=0.0, max_value=200.0, value=55.0)
    О_белок = st.number_input("**Общий белок (г/л)**", min_value=0.0, max_value=20.0, value=6.0)

with col2:
    if st.button("Рассчитать Прогноз"):
        # Input features
        input_features = {
            'О.ж.,%': О_ж,
            'Висц.ж,%': Висц_ж,
            'Мыш.м,%': Мыш_м,
            'ИМТ': ИМТ,
            'АЛТ': АЛТ,
            'АСТ': АСТ,
            'ГГТП': ГГТП,
            'О.белок': О_белок
        }
        probability, predicted_class = predict_probability_and_class(input_features)

        # Display the results with visuals
        st.subheader("**Результаты Прогноза:**")
        st.write(f"**Вероятность:** {probability:.4f}")
        st.write(f"**Класс:** {predicted_class}")

        if predicted_class == "Болен":
            st.error("Модель предсказывает, что вы больны.")
        else:
            st.success("Модель предсказывает, что вы здоровы.")

        # Visual comparison of inputs vs normal ranges
        st.subheader("**Сравнение введенных значений с нормальными диапазонами**")
        fig, ax = plt.subplots(figsize=(8, 5))
        features = ['Ожирение, %', 'Висцеральный жир, %', 'Мышечная масса, %', 'ИМТ', 'АЛТ (ед/л)', 'АСТ (ед/л)', 'ГГТП (ед/л)', 'Общий белок (г/л)']
        user_values = [О_ж, Висц_ж, Мыш_м, ИМТ, АЛТ, АСТ, ГГТП, О_белок]
        normal_min = [normal_ranges[feat][0] for feat in features]
        normal_max = [normal_ranges[feat][1] for feat in features]

        # Ensure x and y are the same size by plotting for every feature
        # Plot normal ranges as horizontal lines
        for i, (min_val, max_val) in enumerate(zip(normal_min, normal_max)):
            ax.plot([min_val, max_val], [i, i], color='gray', lw=6, alpha=0.5, label='Нормальный диапазон' if i == 0 else "")

        # Plot user values as blue markers on top of the normal ranges
        ax.scatter(user_values, range(len(features)), color='blue', s=100, zorder=5, label='Ваши значения')

        # Design improvements
        ax.set_xlabel('Значения', fontsize=12, fontweight='bold')
        ax.set_title('Сравнение показателей с нормальными диапазонами', fontsize=14, fontweight='bold')

        # Customize ticks and labels
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', labelsize=10)
        ax.xaxis.label.set_size(12)

        # Remove right and top spines (axes)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Bold the axes
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        # Set grid for better readability
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)

        # Add a legend with better positioning
        ax.legend(loc='lower right', fontsize=10)

        # Display the plot
        st.pyplot(fig)

# Footer
st.write("---")
st.write("Это приложение создано для помощи в прогнозе НАЖБП на основе различных показателей.")
st.write("Приложение использует биохимические маркеры, индекс массы тела, и другие параметры.")
st.write("Приложение НЕ создано для самостоятельной постановки диагноза.")
    