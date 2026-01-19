🏗️ ASA-PREDICTION MODEL
### AI-Driven Decision Support System for Eco-Efficient Concrete Mix Design

**Developed by:** Aya Mohammed Sanad Aboud  
**Affiliation:** Structural Engineering Dept. | Mansoura University  
**Focus:** Sustainability, Machine Learning, and Multi-Criteria Decision Making (MCDM)

---

## 📝 Project Overview
The **ASA-PREDICTION MODEL** is an advanced analytical platform designed to bridge the gap between AI and sustainable construction. It utilizes a **Multi-output Random Forest** architecture to predict the technical, environmental, and economic performance of concrete mixes containing recycled and eco-friendly materials.
## 🔗 Live Demo
Check out the live application here: [ASA-PREDICTION MODEL App](رابط_تطبيقك_هنا)
## 🚀 Key Features
- **Predictor:** Real-time estimation of 17 performance indicators (Strength, Durability, CO2, Cost).
- **Optimizer:** Identifies the top 5 "Green Mixes" based on laboratory-validated data.
- **Eco-Efficiency Analysis:** Radar charts to visualize the balance between Technical, Environmental, and Economic scores.
- **Model Validation:** Transparent display of $R^2$, RMSE, and MAE metrics.
- **Live Database Integration:** Real-time logging of predictions and user feedback via Google Sheets.

## 📊 Technical Specifications
- **Algorithm:** Random Forest Regression (Multi-output).
- **Dataset:** Diamond Meta-Dataset (1,262 lab samples).
- **Predictive Power:** Validated with an average $R^2$ of **0.925**.
- **Inputs:** 11 raw material parameters (Cement, Water, RCA, Fibers, etc.).

## 🛠️ Tech Stack
- **Backend:** Python 3.12
- **UI Framework:** Streamlit
- **ML Library:** Scikit-learn
- **Data Viz:** Plotly & Seaborn
- **Storage:** Google Sheets API (via streamlit-gsheets)

## 📂 Repository Structure
- `asa_prediction_fixed.py`: Main application code.
- `concrete_model_multi.joblib`: Trained Random Forest model.
- `scaler_multi.joblib`: Pre-processing scaler.
- `requirements.txt`: Python dependencies.
- `cs_validation.png`: Model validation plot (Actual vs. Predicted).

## ⚖️ Disclaimer
This model is for research and guidance purposes. Experimental verification is mandatory for structural applications.

---
© 2026 Aya Mohamed Sanad | Mansoura University
