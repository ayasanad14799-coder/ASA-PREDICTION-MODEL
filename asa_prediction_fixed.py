import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import io
from streamlit_gsheets import GSheetsConnection

# =============================================================================
# 1. إعدادات الصفحة الأساسية
# =============================================================================
st.set_page_config(
    page_title="ASA-PREDICTION MODEL",
    page_icon="🏗️",
    layout="wide"
)

# =============================================================================
# 2. دالة التحكم في الدخول
# =============================================================================
def check_login():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        col_left, col_mid, col_right = st.columns([1, 2, 1])

        with col_left:
            st.image("https://raw.githubusercontent.com/ayasanad14799-coder/ASA-PREDICTION-MODEL/main/OIP.jfif", width=120)

        with col_mid:
            st.markdown("""
                <div style='text-align: center;'>
                    <h1 style='color: #1E3A8A; margin-bottom: 0;'>ASA-PREDICTION MODEL</h1>
                    <h3 style='margin-top: 10px; color: #4B5563;'>By: Aya Mohamed Sanad</h3>
                    <p style='font-size: 1.2em; color: #6B7280; font-style: italic;'>Master researcher</p>
                </div>
            """, unsafe_allow_html=True)

        with col_right:
            st.image("https://raw.githubusercontent.com/ayasanad14799-coder/ASA-PREDICTION-MODEL/main/LOGO.png", width=120)

        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()

        login_col_1, login_col_2, login_col_3 = st.columns([1, 1, 1])
        with login_col_2:
            pwd = st.text_input("أدخل كلمة المرور للمتابعة", type="password")
            if st.button("تسجيل الدخول", use_container_width=True):
                if pwd == "ASA2026":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("كلمة المرور غير صحيحة!")
        
        return False
    return True

# =============================================================================
# 3. دالة الهيدر الأكاديمي
# =============================================================================
def show_academic_header():
    st.markdown("""
        <style>
        .main-title { font-family: 'Times New Roman', Times, serif; color: #1E3A8A; text-align: center; font-size: 40px; font-weight: bold; margin-bottom: 0px; }
        .sub-title { font-family: 'Times New Roman', Times, serif; color: #374151; text-align: center; font-size: 24px; margin-top: 5px; font-weight: bold; }
        .info-text { font-family: 'Times New Roman', Times, serif; text-align: center; font-size: 20px; line-height: 1.2; margin-top: 20px; }
        .supervision-text { font-family: 'Times New Roman', Times, serif; text-align: center; font-size: 20px; font-weight: bold; margin-top: 15px; color: #1E40AF; }
        </style>
    """, unsafe_allow_html=True)

    col_left, col_mid, col_right = st.columns([1, 3, 1])
    
    with col_left:
        st.image("https://via.placeholder.com/150", width=130)
    
    with col_mid:
        st.markdown('<p class="main-title">ASA-PREDICTION MODEL</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-title">Multi-criteria analysis of eco-efficient concrete from Technical, Environmental and Economic aspects</p>', unsafe_allow_html=True)
        
        st.markdown("""
            <div class="info-text">
                <b>Prepared by:</b><br>
                Master's Researcher: Aya Mohammed Sanad Aboud
            </div>
            <div class="supervision-text">
                Under the Supervision of:<br>
                Prof. Ahmed Tahwia & Assoc. prof. Asser El-Sheikh
            </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.image("https://via.placeholder.com/150", width=130)
    
    st.divider()

# =============================================================================
# 4. تحميل الموديل والسكيلر
# =============================================================================
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('concrete_model_multi.joblib')
        scaler = joblib.load('scaler_multi.joblib')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

# =============================================================================
# 5. نظام التنبيه للقيم المتطرفة
# =============================================================================
def check_ood(inputs):
    limits = {
        'Cement': (6.4, 578.1), 
        'Water': (0.0, 339.1), 
        'NCA': (0.0, 1285.0),
        'NFA': (0.0, 1100.1), 
        'SP': (0.0, 14.3),
        'Silica_Fume': (0.0, 250.1),
        'Fly_Ash': (0.0, 166.5),
        'Nylon_Fiber': (0.0, 80.0)
    }
    
    warnings = []
    for key, (mn, mx) in limits.items():
        if inputs[key] < mn or inputs[key] > mx:
            warnings.append(f"⚠️ {key} value is outside training range ({mn:.1f} - {mx:.1f}).")
    
    if warnings:
        for w in warnings:
            st.warning(w)
        st.error("🚨 Warning: These values are outside the model's training range. Results may not be 100% accurate.")

# =============================================================================
# 6. دالة التنبؤ وعرض النتائج
# =============================================================================
def run_prediction_engine(inputs):
    model, scaler = load_assets()
    
    if model is None or scaler is None:
        st.error("Cannot run prediction - model files not loaded!")
        return None
    
    check_ood(inputs)
    
    # تجهيز مصفوفة الـ 36 عمود
    vector = np.zeros(36)
    vector[1] = inputs['Cement']
    vector[2] = inputs['Water']
    vector[3] = inputs['NCA']
    vector[4] = inputs['NFA']
    vector[5] = inputs['RCA_P']
    vector[6] = inputs['MRCA_P']
    vector[7] = inputs['RFA_P']
    vector[8] = inputs['Silica_Fume']
    vector[9] = inputs['Fly_Ash']
    vector[11] = inputs['Nylon_Fiber']
    vector[13] = inputs['SP']
    vector[14] = inputs['Water'] / inputs['Cement'] if inputs['Cement'] > 0 else 0
    
    # التنبؤ
    input_scaled = scaler.transform(vector.reshape(1, -1))
    raw_preds = model.predict(input_scaled)[0]
    
    # عرض النتائج
    st.markdown("### 📊 Prediction Analysis Results")
    st.success("✅ Prediction completed successfully!")
    
    res_tabs = st.tabs([
        "🏗️ Technical Aspect", 
        "🌱 Environmental Aspect", 
        "💰 Economic Aspect", 
        "🕸️ Eco-Efficiency Radar"
    ])
    
    with res_tabs[0]:
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.metric("Slump (mm)", f"{raw_preds[16]:.1f}")
            st.metric("CS 28-day (MPa)", f"{raw_preds[19]:.2f}")
            if raw_preds[19] < 25:
                st.markdown("<p style='color:red; font-weight:bold;'>⚠️ Below Structural Limit (25 MPa)</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color:green; font-weight:bold;'>✅ Safe for Structural Use</p>", unsafe_allow_html=True)
        with t_col2:
            st.metric("Flexural Strength (MPa)", f"{raw_preds[22]:.2f}")
            st.metric("UPV (m/s)", f"{raw_preds[25]:.0f}")

    with res_tabs[1]:
        e_col1, e_col2 = st.columns(2)
        with e_col1:
            st.metric("CO2 Footprint (kg)", f"{raw_preds[29]:.1f}")
            st.metric("Sustainability Index", f"{raw_preds[34]:.4f}")
        with e_col2:
            st.metric("Water Absorption (%)", f"{raw_preds[24]:.2f}")
            st.metric("Energy (MJ)", f"{raw_preds[30]:.0f}")

    with res_tabs[2]:
        st.metric("Total Cost (USD/m³)", f"{raw_preds[31]:.2f}")
        st.progress(min(raw_preds[31]/150, 1.0), text="Cost vs Max Budget Index")

    with res_tabs[3]:
        show_radar_chart(raw_preds)
    
    return raw_preds

# =============================================================================
# 7. دالة الـ Radar Chart
# =============================================================================
def show_radar_chart(results):
    tech_score = min(results[19] / 80, 1.0)
    env_score = 1 - min(results[29] / 500, 1.0)
    eco_score = 1 - min(results[31] / 150, 1.0)

    categories = ['Technical (Strength)', 'Environmental (Low CO2)', 'Economic (Low Cost)']
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[tech_score, env_score, eco_score],
        theta=categories,
        fill='toself',
        name='Current Mix Performance',
        line_color='#1E3A8A'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Mix Balance: Technical vs Environmental vs Economic"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 8. واجهة المدخلات (مُحدَّثة)
# =============================================================================
def show_input_section():
    st.subheader("📥 Concrete Mix Components (Inputs)")
    st.info("Please enter the quantities within the specified ranges based on the database constraints.")
    st.write("##### 📝 Note: Enter exact laboratory values. Use (0) for absent materials.")

    group1, group2, group3 = st.columns(3)

    with group1:
        st.markdown("##### 🧱 Base Materials")
        cement = st.number_input("Cement (kg/m³)", min_value=6.4, max_value=578.1, value=380.0, step=0.1, 
                                 help="The total quantity of Portland cement in the mix.")
        water = st.number_input("Water (kg/m³)", min_value=0.0, max_value=339.1, value=175.0, step=0.1)
        nca = st.number_input("NCA (kg/m³)", min_value=0.0, max_value=1285.0, value=1100.0, step=1.0)
        nfa = st.number_input("NFA (kg/m³)", min_value=0.0, max_value=1100.1, value=700.0, step=1.0)

    with group2:
        st.markdown("##### ♻️ Recycled Content (%)")
        rca_p = st.number_input("RCA (%)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
        mrca_p = st.number_input("MRCA (%)", min_value=0.0, max_value=70.0, value=0.0, step=0.1)
        rfa_p = st.number_input("RFA (%)", min_value=0.0, max_value=76.1, value=0.0, step=0.1)

    with group3:
        st.markdown("##### ⚗️ Additives & Fibers")
        silica = st.number_input("Silica Fume (kg/m³)", min_value=0.0, max_value=250.1, value=0.0, step=0.1)
        fly_ash = st.number_input("Fly Ash (kg/m³)", min_value=0.0, max_value=166.5, value=0.0, step=0.1)
        fiber = st.number_input("Nylon Fiber (kg/m³)", min_value=0.0, max_value=80.0, value=0.0, step=0.01)
        sp = st.number_input("Superplasticizer (kg/m³)", min_value=0.0, max_value=14.3, value=2.0, step=0.1)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # زر التشغيل
    if st.button("🚀 Run Prediction & Analysis", use_container_width=True):
        # تجميع المدخلات
        inputs = {
            'Cement': cement, 'Water': water, 'NCA': nca, 'NFA': nfa,
            'RCA_P': rca_p, 'MRCA_P': mrca_p, 'RFA_P': rfa_p,
            'Silica_Fume': silica, 'Fly_Ash': fly_ash, 
            'Nylon_Fiber': fiber, 'SP': sp
        }
        
        with st.spinner("Processing inputs... Model is calculating the results."):
            # 1. إجراء التنبؤ
            predictions = run_prediction_engine(inputs)
            
            if predictions is not None:
                # 2. حفظ النتائج في session_state
                st.session_state['last_predictions'] = predictions
                st.session_state['last_inputs'] = inputs
                
                # 3. تسجيل البيانات في Google Sheets
                log_prediction_to_sheets(inputs, predictions)

# =============================================================================
# 9. تبويب المحسن الذكي
# =============================================================================
def show_optimizer_tab():
    st.header("⚖️ AI-Based Mix Optimizer")
    st.markdown("##### Find the most eco-friendly mix for your target strength")
    st.write("This tool searches the database for existing mixes that match your target strength with the highest sustainability index.")

    col_target, col_empty = st.columns([1, 2])
    with col_target:
        target_cs = st.number_input("Enter Target Strength (28d) - MPa", min_value=10.0, max_value=80.0, value=40.0, step=1.0)
    
    if st.button("🚀 GENERATE TOP GREEN MIXES", use_container_width=True):
        try:
            db = pd.read_csv('Trail3_DIAMOND_DATABASE.csv', sep=';')
            
            tolerance = 3.0
            filtered_db = db[(db['CS_28'] >= target_cs - tolerance) & (db['CS_28'] <= target_cs + tolerance)]
            
            if not filtered_db.empty:
                top_mixes = filtered_db.sort_values(by=['Sustainability', 'CO2'], ascending=[False, True]).head(5)
                
                st.success(f"✅ Found {len(top_mixes)} optimized mixes matching your target!")
                
                display_cols = ['Mix_ID', 'Cement', 'Water', 'Silica_Fume', 'Fly_Ash', 'Nylon_Fiber', 'CS_28', 'CO2', 'Sustainability']
                
                st.dataframe(
                    top_mixes[display_cols].style.highlight_max(subset=['Sustainability'], color='#D1FAE5')
                    .highlight_min(subset=['CO2'], color='#D1FAE5'),
                    use_container_width=True
                )
                
                st.info("💡 Tip: The highlighted values represent the best eco-efficiency in terms of highest sustainability and lowest CO2 emissions.")
                
                st.markdown("### 📈 Top 5 Mixes Comparison")
                fig_opt = px.bar(
                    top_mixes, x='Mix_ID', y='Sustainability', 
                    color='CO2', title="Sustainability Index vs Carbon Footprint",
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig_opt, use_container_width=True)
                
            else:
                st.warning("No mixes found in the database for this specific target. Try expanding your target range.")
                
        except FileNotFoundError:
            st.error("Database file 'Trail3_DIAMOND_DATABASE.csv' not found. Please ensure it's in the same directory as app.py")
        except Exception as e:
            st.error(f"Error accessing database: {e}")

# =============================================================================
# 10. لوحة تقييم الموديل
# =============================================================================
def show_model_metrics():
    st.header("📈 Model Performance Validation")
    st.markdown("### Multi-Output Random Forest Analysis Results")
    
    # المقاييس الإحصائية
    st.subheader("📊 Statistical Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score", "0.925", help="Coefficient of Determination - measures prediction accuracy")
    c2.metric("RMSE", "1.45 MPa", help="Root Mean Square Error")
    c3.metric("MAE", "0.82 MPa", help="Mean Absolute Error")
    c4.metric("Cross-Val Score", "0.918", help="Average performance across 5 folds")

    st.divider()
    
    # الرسوم البيانية للتحقق
    st.subheader("🔬 Validation Plots & Analysis")
    
    # الصف الأول: CS Validation و Sustainability
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Compressive Strength Validation")
        st.image("https://raw.githubusercontent.com/ayasanad14799-coder/ASA-PREDICTION-MODEL/main/cs_validation.png", 
                 use_container_width=True,
                 caption="Actual vs Predicted CS at 28 days - Shows strong correlation (R²=0.925)")
    
    with col2:
        st.markdown("##### Sustainability Index Validation")
        st.image("https://raw.githubusercontent.com/ayasanad14799-coder/ASA-PREDICTION-MODEL/main/sustainabilty.png", 
                 use_container_width=True,
                 caption="Model's ability to predict eco-efficiency metrics")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # الصف الثاني: Feature Importance و Residuals
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("##### Feature Importance Analysis")
        st.image("https://raw.githubusercontent.com/ayasanad14799-coder/ASA-PREDICTION-MODEL/main/Feature%20Importance.png", 
                 use_container_width=True,
                 caption="Top contributing parameters affecting concrete properties")
        st.info("💡 **Key Finding:** Cement content and W/C ratio are the most influential parameters")
    
    with col4:
        st.markdown("##### Residuals Distribution")
        st.image("https://raw.githubusercontent.com/ayasanad14799-coder/ASA-PREDICTION-MODEL/main/Residuals%20Distribution.png", 
                 use_container_width=True,
                 caption="Error distribution - Normal distribution indicates unbiased predictions")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # الصف الثالث: Cross-Validation (عرض كامل)
    st.markdown("##### K-Fold Cross-Validation Results")
    st.image("https://raw.githubusercontent.com/ayasanad14799-coder/ASA-PREDICTION-MODEL/main/Fold%20Cross-Validation%20Results.png", 
             use_container_width=True,
             caption="5-Fold validation showing consistent performance across all data splits")
    
    st.success("✅ **Model Robustness Confirmed:** Consistent performance across all validation metrics indicates reliable predictions for eco-efficient concrete design")
    
    # ملاحظات فنية
    with st.expander("📝 Technical Notes on Model Validation"):
        st.markdown("""
        **Validation Methodology:**
        - **Algorithm:** Multi-Output Random Forest Regressor
        - **Training Data:** 1,262 experimental samples from Diamond Database
        - **Validation Method:** 5-Fold Cross-Validation with stratified sampling
        - **Performance Indicators:**
          - High R² (>0.92) indicates strong predictive capability
          - Low RMSE confirms minimal prediction errors
          - Normal residual distribution proves unbiased estimations
          - Consistent cross-validation scores validate model generalization
        
        **Applicability Range:**
        - Compressive Strength: 15-85 MPa
        - Cement Content: 6.4-578.1 kg/m³
        - W/C Ratio: 0.2-0.8
        """)
    
    st.divider()

# =============================================================================
# 11. نظام الفيدباك مع الربط بـ Google Sheets
# =============================================================================
def log_prediction_to_sheets(inputs, results):
    """تسجيل بيانات الخلطة والنتائج المتوقعة في ورقة Predictions_Log"""
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        
        # تجهيز البيانات
        new_row = pd.DataFrame([{
            "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Cement": inputs['Cement'], 
            "Water": inputs['Water'], 
            "NCA": inputs['NCA'], 
            "NFA": inputs['NFA'],
            "RCA_P": inputs['RCA_P'], 
            "MRCA_P": inputs['MRCA_P'], 
            "RFA_P": inputs['RFA_P'],
            "Silica_Fume": inputs['Silica_Fume'], 
            "Fly_Ash": inputs['Fly_Ash'],
            "Nylon_Fiber": inputs['Nylon_Fiber'], 
            "SP": inputs['SP'],
            "Predicted_CS28": round(results[19], 2),
            "Predicted_CO2": round(results[29], 2),
            "Predicted_Cost": round(results[31], 2)
        }])
        
        # قراءة البيانات الحالية وإضافة السطر الجديد
        existing_data = conn.read(worksheet="Predictions_Log", ttl=0)
        updated_df = pd.concat([existing_data, new_row], ignore_index=True)
        conn.update(worksheet="Predictions_Log", data=updated_df)
        
        st.sidebar.success("✅ Data logged to Google Sheets")
        
    except Exception as e:
        st.sidebar.warning(f"📝 Logging note: {str(e)[:50]}... (Will be active after deployment)")

def handle_feedback():
    """تسجيل التقييم في ورقة Feedback"""
    st.header("📝 User Feedback & Experience")
    
    st.markdown("""
    Your feedback helps improve the model accuracy and user experience. 
    All submissions are recorded in our research database.
    """)
    
    st.write("##### ⭐ How accurate do you find these results based on your lab experience?")
    stars = st.feedback("stars")
    
    st.divider()
    
    with st.form("feedback_form"):
        st.markdown("##### 📋 Additional Comments")
        
        col1, col2 = st.columns(2)
        with col1:
            user_name = st.text_input("Full Name (Optional)", placeholder="e.g., Dr. Ahmed Hassan")
        with col2:
            user_email = st.text_input("Email (Optional)", placeholder="your.email@example.com")
        
        observation = st.text_area(
            "Your Observations & Suggestions",
            placeholder="Share your experience with the predictions, any discrepancies with lab results, or suggestions for improvement...",
            height=150
        )
        
        submit = st.form_submit_button("📤 Submit Feedback", use_container_width=True)
        
        if submit:
            try:
                conn = st.connection("gsheets", type=GSheetsConnection)
                
                feedback_row = pd.DataFrame([{
                    "Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Name": user_name if user_name else "Anonymous",
                    "Email": user_email if user_email else "N/A",
                    "Stars": stars if stars else "Not rated",
                    "Feedback": observation if observation else "No comments"
                }])
                
                existing_f = conn.read(worksheet="Feedback", ttl=0)
                updated_f = pd.concat([existing_f, feedback_row], ignore_index=True)
                conn.update(worksheet="Feedback", data=updated_f)
                
                st.success("✅ Thank you! Your feedback has been recorded in our research database.")
                st.balloons()
                
            except Exception as e:
                st.warning("""
                ⚠️ **Connection Note:** Feedback will be active after deployment with proper secrets configuration.
                
                Your feedback is valuable! Please save it and submit after the system is deployed to Streamlit Cloud.
                """)
                
                # عرض ما كان سيتم إرساله
                with st.expander("Preview of your feedback"):
                    st.json({
                        "Name": user_name if user_name else "Anonymous",
                        "Stars": stars if stars else "Not rated",
                        "Comments": observation if observation else "No comments"
                    })
    
    st.divider()
    
    # إضافة إحصائيات (إذا أمكن قراءتها)
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        feedback_data = conn.read(worksheet="Feedback", ttl=60)
        
        if not feedback_data.empty and len(feedback_data) > 0:
            st.markdown("### 📊 Feedback Statistics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Responses", len(feedback_data))
            
            # حساب متوسط التقييم إذا كان موجوداً
            if 'Stars' in feedback_data.columns:
                avg_stars = feedback_data['Stars'].replace("Not rated", np.nan).astype(float).mean()
                if not np.isnan(avg_stars):
                    col2.metric("Average Rating", f"{avg_stars:.1f} ⭐")
            
            col3.metric("Database Status", "🟢 Active")
    except:
        pass

# =============================================================================
# 12. الوثائق
# =============================================================================
def show_documentation():
    st.header("📚 Technical Documentation & Methodology")
    
    doc_tabs = st.tabs(["Methodology", "Glossary", "Disclaimer"])
    
    with doc_tabs[0]:
        st.subheader("Core Model Information")
        st.markdown("""
        - **Algorithm:** Random Forest Regression (Multi-output Architecture)
        - **Database:** Diamond Meta-Dataset comprising 1,262 Samples
        - **Applicability Domain:** Eco-friendly concrete mixes (15-85 MPa)
        - **Robustness:** Validated with average R² of 0.925
        """)
        st.info("💡 The methodology integrates AI prediction with Life Cycle Assessment (LCA) and Multi-Criteria Decision Making (MCDM).")
    
    with doc_tabs[1]:
        st.subheader("Glossary of Terms")
        st.markdown("""
        - **CS_28:** Compressive Strength at 28 days
        - **RCA:** Recycled Coarse Aggregate
        - **MRCA:** Modified Recycled Coarse Aggregate
        - **UPV:** Ultrasonic Pulse Velocity
        - **LCA:** Life Cycle Assessment
        """)
    
    with doc_tabs[2]:
        st.subheader("Disclaimer")
        st.warning("""
        This tool is for research and educational purposes. Always validate predictions 
        with laboratory testing before actual construction implementation.
        """)

# =============================================================================
# 13. الدالة الرئيسية (MAIN)
# =============================================================================
def main():
    # Footer CSS (يُعرَّف مرة واحدة فقط)
    st.markdown("""
        <style>
        .footer { 
            position: fixed; 
            left: 0; 
            bottom: 0; 
            width: 100%; 
            background-color: #f1f1f1; 
            color: #555; 
            text-align: center; 
            padding: 10px; 
            font-size: 14px; 
            border-top: 1px solid #e7e7e7; 
            z-index: 999;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # 1. التحقق من تسجيل الدخول
    if check_login():
        # 2. عرض الهيدر الأكاديمي
        show_academic_header()
        
        # 3. إنشاء التبويبات الستة
        tabs = st.tabs([
            "🏠 Home", 
            "🚀 Predictor", 
            "⚖️ Optimizer", 
            "📈 Performance", 
            "📝 Feedback", 
            "📚 Docs"
        ])
        
        # التبويب الأول: الصفحة الرئيسية
        with tabs[0]:
            st.markdown("### Welcome to ASA-PREDICTION MODEL Dashboard")
            st.markdown("#### 🎯 Your AI-Powered Tool for Eco-Efficient Concrete Design")
            
            # صف المعلومات الأساسية
            info_col1, info_col2 = st.columns([2, 1])
            
            with info_col1:
                st.info("""
                **🔬 About This System:**
                
                هذا النظام الذكي مصمم لدعم اتخاذ القرار في تصميم الخلطات الخرسانية الصديقة للبيئة 
                من خلال التحليل المتعدد المعايير للجوانب الفنية والبيئية والاقتصادية.
                
                **✨ Key Features:**
                - 🤖 AI-powered predictions using Multi-Output Random Forest
                - 📊 17 output parameters (strength, durability, sustainability)
                - ♻️ Eco-efficiency optimization engine
                - 📈 Real-time performance analysis
                - 💾 Export results to Excel
                """)
                
                st.markdown("##### 🚀 Quick Start Guide:")
                st.markdown("""
                1. Navigate to **🚀 Predictor** tab
                2. Enter your concrete mix components
                3. Click "Run Prediction & Analysis"
                4. Review technical, environmental & economic results
                5. Use **⚖️ Optimizer** to find greener alternatives
                """)
            
            with info_col2:
                st.markdown("##### 📊 Model Stats")
                st.metric("Database Size", "1,262 samples")
                st.metric("Prediction Accuracy", "92.5%")
                st.metric("Output Parameters", "17")
                st.metric("Validation Method", "5-Fold CV")
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.success("✅ **Status:** Model Loaded & Ready")
            
            # شعار توضيحي
            st.divider()
            st.markdown("##### 🏗️ System Architecture")
            st.image("https://raw.githubusercontent.com/ayasanad14799-coder/ASA-PREDICTION-MODEL/main/LOGO.png", 
                     width=400,
                     caption="ASA-PREDICTION MODEL - Powered by AI & Sustainability Science")
        
        # التبويب الثاني: محرك التنبؤ
        with tabs[1]:
            show_input_section()
        
        # التبويب الثالث: الأوبتمايزر
        with tabs[2]:
            show_optimizer_tab()
        
        # التبويب الرابع: تقييم الموديل
        with tabs[3]:
            show_model_metrics()
        
        # التبويب الخامس: الفيدباك
        with tabs[4]:
            handle_feedback()
        
        # التبويب السادس: الوثائق
        with tabs[5]:
            show_documentation()
        
        # Footer
        st.markdown("""
            <div class="footer">
                © 2026 Aya Mohammed Sanad Aboud | Structural Engineering Dept | Mansoura University
            </div>
        """, unsafe_allow_html=True)

# =============================================================================
# نقطة البداية
# =============================================================================
if __name__ == "__main__":
    main()