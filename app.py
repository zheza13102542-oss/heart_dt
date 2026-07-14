import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# Custom CSS - Minimalist Design
# ============================================================
st.markdown("""
<style>
    /* พื้นหลังหลัก */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8edf3 100%);
    }
    
    /* หัวข้อหลัก */
    .main-title {
        text-align: center;
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    
    .sub-title {
        text-align: center;
        color: #7f8c8d;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* การ์ดผลลัพธ์ */
    .result-card {
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    .result-safe {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 6px solid #28a745;
    }
    
    .result-danger {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 6px solid #dc3545;
    }
    
    .result-icon {
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
    }
    
    .result-text {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    .result-prob {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    /* ปุ่มหลัก */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 50px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Input section */
    .input-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    /* ซ่อน footer ของ Streamlit */
    footer {visibility: hidden;}
    
    /* ปรับแต่ง slider และ input */
    .stSlider > div > div > div {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Load Model
# ============================================================
@st.cache_resource
def load_model():
    package = joblib.load('heart_disease_model.pkl')
    return package['pipeline'], package['model'], package['feature_names']

pipeline, model, feature_names = load_model()

# ============================================================
# Header Section
# ============================================================
st.markdown('<p class="main-title">❤️ Heart Disease Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">ระบบประเมินความเสี่ยงโรคหัวใจด้วย Machine Learning</p>', unsafe_allow_html=True)

# ============================================================
# Input Form
# ============================================================
st.markdown("### 📋 กรอกข้อมูลสุขภาพ")

# แบ่งเป็น 2 คอลัมน์
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    age = st.slider("🎂 อายุ (ปี)", 20, 90, 50)
    sex = st.selectbox("⚧️ เพศ", options=[1, 0], 
                       format_func=lambda x: "ชาย (Male)" if x == 1 else "หญิง (Female)")
    cp = st.selectbox("💔 อาการเจ็บหน้าอก", 
                      options=[3, 1, 2, 4], 
                      format_func=lambda x: {
                          3: "ASY - ไม่มีอาการ",
                          1: "ATA - เจ็บหน้าอกไม่典型", 
                          2: "NAP - เจ็บหน้าอกไม่ใช่ Angina",
                          4: "TA - เจ็บหน้าอก典型"
                      }[x])
    trestbps = st.number_input("🩸 ความดันโลหิตขณะพัก (mmHg)", 80, 200, 120)
    chol = st.number_input("🧪 คลอเรสเตอรอล (mg/dl)", 100, 600, 200)
    fbs = st.selectbox("🍬 น้ำตาลในเลือด > 120 mg/dl", options=[0, 1], 
                       format_func=lambda x: "ไม่ (No)" if x == 0 else "ใช่ (Yes)")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    restecg = st.selectbox("📈 ผล ECG ขณะพัก", options=[0, 1, 2],
                          format_func=lambda x: {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"}[x])
    thalach = st.number_input("💓 อัตราเต้นหัวใจสูงสุด (bpm)", 60, 210, 140)
    exang = st.selectbox("🏃 เจ็บหน้าอกขณะออกกำลังกาย", options=[0, 1],
                        format_func=lambda x: "ไม่ (No)" if x == 0 else "ใช่ (Yes)")
    oldpeak = st.number_input("📉 ค่า ST Depression", -3.0, 7.0, 1.0, step=0.1)
    slope = st.selectbox("📊 ความชัน ST Segment", options=[2, 1, 3],
                        format_func=lambda x: {1: "Up", 2: "Flat", 3: "Down"}[x])
    
    st.markdown('</div>', unsafe_allow_html=True)

# รวบรวมข้อมูล
input_data = {
    'Age': age, 'Sex': sex, 'ChestPainType': cp,
    'RestingBP': trestbps, 'Cholesterol': chol, 'FastingBS': fbs,
    'RestingECG': restecg, 'MaxHR': thalach, 'ExerciseAngina': exang,
    'Oldpeak': oldpeak, 'ST_Slope': slope
}
input_df = pd.DataFrame([input_data])

# ============================================================
# Predict Button
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 ทำนายความเสี่ยง", type="primary"):
    with st.spinner("🧠 กำลังวิเคราะห์ข้อมูล..."):
        time.sleep(0.8)
        
        # เรียงคอลัมน์ให้ตรงกับตอนเทรน
        input_df = input_df[feature_names]
        
        # Transform และ Predict
        input_transformed = pipeline.transform(input_df)
        prediction = model.predict(input_transformed)[0]
        probability = model.predict_proba(input_transformed)[0]
    
    # แสดงผลลัพธ์
    st.markdown("<br>", unsafe_allow_html=True)
    
    if prediction == 1:
        st.markdown(f'''
        <div class="result-card result-danger">
            <div class="result-icon">⚠️</div>
            <div class="result-text" style="color: #721c24;">พบความเสี่ยงโรคหัวใจ</div>
            <div class="result-prob" style="color: #dc3545;">{probability[1]*100:.1f}%</div>
            <p style="color: #721c24; margin: 0;">ความน่าจะเป็นที่จะเป็นโรค</p>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="result-card result-safe">
            <div class="result-icon">✅</div>
            <div class="result-text" style="color: #155724;">ไม่พบความเสี่ยงโรคหัวใจ</div>
            <div class="result-prob" style="color: #28a745;">{probability[0]*100:.1f}%</div>
            <p style="color: #155724; margin: 0;">ความน่าจะเป็นที่จะไม่เป็นโรค</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # คำแนะนำ
    st.info("""
    💡 **คำแนะนำ:** ผลการทำนายนี้เป็นเพียงการประเมินเบื้องต้นจากโมเดล Machine Learning เท่านั้น 
    ไม่สามารถใช้แทนการวินิจฉัยจากแพทย์ผู้เชี่ยวชาญได้ หากท่านมีอาการผิดปกติ กรุณาปรึกษาแพทย์ทันที
    """)

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #95a5a6; font-size: 0.9rem;">'
    'Developed with ❤️ using Python, Scikit-learn & Streamlit | © 2026'
    '</p>', 
    unsafe_allow_html=True
)