import streamlit as st

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Wildlife Habitat Project", page_icon="🐾", layout="wide")

# 2. ฟังก์ชันสำหรับหน้า Project Info
def project_info_page():
    st.title("📂 ข้อมูลโปรเจค (Project Info)")
    st.write("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔬 ระบบ Ensemble Model (Stacking)")
        st.markdown("""
        **รายละเอียดโมเดล (อ้างอิง animal.ipynb):**
        * **Algorithms:** ใช้เทคนิค **Stacking Ensemble** โดยมี `Random Forest` และ `XGBoost` เป็นโมเดลหลัก และใช้ `Logistic Regression` เป็นตัวสรุปผล
        * **การเตรียมข้อมูล:** มีการใช้ `StandardScaler` และเทคนิค **SMOTE** เพื่อจัดการกับปัญหาข้อมูลไม่สมดุล (Imbalanced Data)
        * **แหล่งข้อมูล (Dataset Source):**
          * [Animal Kingdom Taxonomy (Kaggle)](https://www.kaggle.com/datasets/hasanarcas/animal-kingdom-taxonomy) 🔗
        """)
        
    with col2:
        st.subheader("🧠 ระบบ Neural Network (ANN)")
        st.markdown("""
        **รายละเอียดโมเดล (อ้างอิง train.ipynb):**
        * **Architecture:** พัฒนาโครงสร้าง **ANN** เองด้วย `3 Hidden Layers` 
        * **Activation:** ใช้ `ReLU` ในการเรียนรู้ และ `Softmax` ในการระบุประเภทคลาส
        * **แหล่งข้อมูล (Dataset Source):**
          * [Fruit and Vegetable Image Recognition (Kaggle)](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition) 🔗 
          *(หมายเหตุ: ใช้สำหรับฝึกฝนโครงข่ายประสาทเทียมในการจำแนกรูปภาพ)*
        """)
    
    st.markdown("---")
    st.info("💡 เลือกหน้าพยากรณ์ที่ Sidebar เพื่อทดสอบประสิทธิภาพของแต่ละโมเดล")

# 3. การตั้งค่า Navigation (ใช้ไฟล์เดิมของคุณ)
pg = st.navigation({
    "เมนูหลัก": [
        st.Page(project_info_page, title="ข้อมูลโปรเจค", icon="📂"),
    ],
    "ระบบพยากรณ์": [
        st.Page("pages/1_Ensemble.py", title="Ensemble", icon="🌳"),
        st.Page("pages/2_NN.py", title="Neural Network", icon="🧠"),
    ]
})

# 4. รันระบบ
pg.run()