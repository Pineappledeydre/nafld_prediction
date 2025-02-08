# NAFLD Prediction App

**NAFLD Prediction App** is a Streamlit-based web application that predicts the probability of Non-Alcoholic Fatty Liver Disease (NAFLD) using a trained Explainable Boosting Machine (EBM) model.

## Features
- **User Input Fields**: Enter various health parameters.
- **Prediction Model**: Uses a trained **EBM model** (`ebm_model.pkl`) for classification.
- **Probability Output**: Displays the likelihood of NAFLD.
- **Visual Comparison**: A graph showing user values vs. normal ranges.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Pineappledeydre/nafld_prediction.git
   cd nafld_prediction
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:


## Files
- `streamlit_app.py` → Main Streamlit app file.
- `models/ebm_model.pkl` → Trained Explainable Boosting Machine (EBM) model.
- `requirements.txt` → List of dependencies.
- `README.md` → Documentation.

## Model Details
- **Algorithm**: Explainable Boosting Machine (EBM).
- **Dataset**: Cleaned medical dataset for NAFLD prediction.
- **Features Used**: Age, BMI, cholesterol, ALT, AST, glucose, etc.
- **Output**: Probability of having NAFLD (`0 = Healthy, 1 = NAFLD`).

## Preview
![App Screenshot](screenshot.png)

## Disclaimer
This application is **not a substitute for professional medical diagnosis**. Always consult a healthcare professional for accurate assessments.

---
Developed with ❤️ by Pineappledeydre
