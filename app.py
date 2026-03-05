import streamlit as st

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Wildlife Habitat Project", page_icon="🐾", layout="wide")

# 2. ฟังก์ชันสำหรับหน้า Project Info
def project_info_page():
    st.title("📂 ข้อมูลรายละเอียดโปรเจค (Detailed Project Info)")
    st.write("อธิบายแนวทางการพัฒนาโมเดลตามข้อกำหนดรายวิชา")
    st.markdown("---")
    
    # แบ่งเป็น 2 Tabs สำหรับ 2 โมเดล ตามที่อาจารย์กำหนด 
    tab1, tab2 = st.tabs(["🌳 โมเดลที่ 1: Ensemble Learning", "🧠 โมเดลที่ 2: Neural Network"])

    with tab1:
        st.header("1. Machine Learning แบบ Ensemble")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📝 รายละเอียดและที่มา")
            st.write("**ชุดข้อมูล (Dataset):** Animal Kingdom Taxonomy จาก Kaggle ")
            st.write("**คุณลักษณะ (Features):** ข้อมูลอนุกรมวิธานของสัตว์เพื่อจำแนกประเภทตามถิ่นที่อยู่ ")
            
            st.subheader("🛠️ การเตรียมข้อมูล (Data Preparation)")
            st.markdown("""
            เพื่อให้เป็นไปตามข้อกำหนดที่ต้องจัดการข้อมูลที่ไม่สมบูรณ์:
            1. **Handling Missing Values:** ตรวจสอบและจัดการค่าว่างในชุดข้อมูล
            2. **Feature Scaling:** ใช้ `StandardScaler` เพื่อปรับช่วงข้อมูลให้เหมาะสม
            3. **Imbalanced Data:** ใช้เทคนิค **SMOTE** เพื่อเพิ่มจำนวนข้อมูลในคลาสที่มีน้อย 
            """)

        with col2:
            st.subheader("🔬 ทฤษฎีและขั้นตอนการพัฒนา ")
            st.markdown("""
            **อัลกอริทึมที่ใช้ (Stacking Ensemble):**
            ใช้การรวมพลังของ 3 โมเดลตามข้อกำหนด:
            1. **Random Forest:** ใช้สร้าง Decision Trees หลายต้นเพื่อลด Overfitting
            2. **XGBoost:** ใช้เทคนิค Gradient Boosting เพื่อเพิ่มความแม่นยำ
            3. **Logistic Regression:** ทำหน้าที่เป็น *Meta-Learner* เพื่อสรุปผลลัพธ์สุดท้าย
            """)
            st.info("ขั้นตอน: นำข้อมูลที่เตรียมไว้เข้าสู่การ Train โมเดลย่อย และใช้ Meta-model ในการทำนายผลลัพธ์สุดท้าย ")

    with tab2:
        st.header("2. Neural Network (ANN)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📝 รายละเอียดและที่มา")
            st.write("**ชุดข้อมูล (Dataset):** Fruit and Vegetable Image Recognition จาก Kaggle ")
            st.write("**ประเภท:** ข้อมูลแบบ Unstructured (รูปภาพ) ")
            
            st.subheader("🛠️ การเตรียมข้อมูล (Data Preparation)")
            st.markdown("""
            1. **Image Resizing:** ปรับขนาดรูปภาพให้เท่ากันทั้งหมด (เช่น 224x224)
            2. **Normalization:** ปรับค่า Pixel ให้อยู่ในช่วง 0-1
            3. **Data Augmentation:** เพิ่มความหลากหลายให้ข้อมูลรูปภาพเพื่อป้องกัน Overfitting
            """)

        with col2:
            st.subheader("🧠 โครงสร้างโมเดล (Architecture) ")
            st.markdown("""
            **การออกแบบโครงสร้าง (Custom ANN):**
            * **Input Layer:** รับข้อมูลรูปภาพที่ผ่านการ Flatten
            * **Hidden Layers:** ออกแบบเองทั้งหมด 3 ชั้น  ใช้ `ReLU` เป็น Activation Function
            * **Output Layer:** ใช้ `Softmax` สำหรับการจำแนกหลายคลาส (Multi-class Classification)
            """)
            st.success("โมเดลนี้ได้รับการออกแบบโครงสร้างเองเพื่อให้เหมาะสมกับลักษณะของรูปภาพในชุดข้อมูล ")

    st.markdown("---")
    st.subheader("🔗 แหล่งอ้างอิง (References)")
    st.write("- Dataset 1: [Animal Kingdom Taxonomy (Kaggle)](https://www.kaggle.com/datasets/hasanarcas/animal-kingdom-taxonomy)")
    st.write("- Dataset 2: [Fruit and Vegetable Recognition (Kaggle)](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)")
    st.write("- Library: Streamlit, Scikit-learn, TensorFlow/Keras, XGBoost")

# 3. การตั้งค่า Navigation
pg = st.navigation({
    "เมนูหลัก": [
        st.Page(project_info_page, title="ข้อมูลโปรเจค", icon="📂"),
    ],
    "ระบบพยากรณ์": [
        st.Page("pages/1_Ensemble.py", title="Ensemble Test", icon="🌳"),
        st.Page("pages/2_NN.py", title="Neural Network Test", icon="🧠"),
    ]
})

# 4. รันระบบ
pg.run()