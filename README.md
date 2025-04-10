Here’s your updated **README** with the requested additions:

---

# **NAFLD Prediction App**

**NAFLD Prediction App** is a **Streamlit-based web application** that predicts the probability of **Non-Alcoholic Fatty Liver Disease (NAFLD)** using a trained **Explainable Boosting Machine (EBM)** model.  

**Key Features**:
- **User Input Form**: Enter personal health data (e.g., ALT, AST, BMI, etc.).
- **Machine Learning Model**: Uses a trained **EBM model** (`ebm_model.pkl`) for prediction.
- **Probability Output**: Displays the likelihood of NAFLD **(0 = Healthy, 1 = NAFLD)**.
- **Visual Comparisons**:  
    **Feature Importance** – Understand which factors drive predictions.  
    **Personalized Graph** – Compares your values with normal health ranges.  

---

## **Installation & Setup**

### **1) Clone the Repository**
```bash
git clone https://github.com/Pineappledeydre/nafld_prediction.git
cd nafld_prediction
```

### **2) Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3) Run the Application**
```bash
streamlit run streamlit_app.py
```

---

## **📂 Project Structure**
```
nafld_prediction
├── models
│   ├── ebm_model.pkl             # Trained EBM model
│   ├── ebm_model_v2.pkl          # Trained EBM model - version 2
├── data
│   ├── feature_min_max_values.csv  # Reference values for normalization
├── streamlit_app.py             # Main Streamlit app
├── requirements.txt             # Dependencies
├── README.md                    # Documentation
└── screenshot.png               # App preview image
```

---

## **Model Details**
- **Algorithm**: Explainable Boosting Machine (EBM) – A transparent, interpretable ML model.
- **Dataset**: Clinical data on liver health markers and metabolic indicators.  
  ➤ **The training dataset can be provided upon request.**
- **Features Used**:
  - **Metabolic Markers**: BMI, LDL, Triglycerides, Glucose, Insulin.
  - **Liver Enzymes**: ALT, AST, GGT.
  - **Inflammatory Markers**: CRP, Ferritin.
  - **Body Composition**: Visceral Fat, Body Fat %, Skeleton %.  

---

## **Visualizations in the App**
**Prediction Results** – Shows the probability of NAFLD based on your inputs.  
**Feature Importance** – Ranks the most influential factors in prediction.  
**Comparison Chart** – Displays where your values fall compared to normal health ranges.  
**Extreme Value Detection** – Highlights any **abnormal health markers** with red warnings.  

---

## **Preview**
🔗 **[Live Streamlit App](https://nafld-prediction.streamlit.app/)** – Try out the prediction tool in real-time.  
🔗 **[Live Dashboard](https://datalens.yandex/ppnglqr1jrjeb)** – Interactive data insights from the NAFLD prediction model.

---

## **Disclaimer**
This application is **not a substitute for professional medical advice**. Please consult a healthcare professional for any health concerns.

---

**Developed with ❤️ by Pineappledeydre**  
If you find this useful, consider giving the repository a ⭐  

---
