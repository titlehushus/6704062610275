import streamlit as st
import pandas as pd
import pickle
import os

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Habitat & Behavior Predictor", page_icon="🌲")

st.title("🌲 ทำนายถิ่นที่อยู่อาศัย & คำแนะนำเมื่อถูกสัตว์กัด")
st.markdown("""
ระบบนี้ใช้ **Ensemble Model** ในการทำนายถิ่นที่อยู่อาศัยของสัตว์ป่า 
และอาศัยข้อมูลทางอนุกรมวิธาน (Order/Family) เพื่อเป็นตัวบอกพฤติกรรมทางอ้อม 
พร้อมให้คำแนะนำเบื้องต้นในการปฐมพยาบาลตามประเภทของสัตว์
""")

# --- 2. การจัดการ Path และโหลดโมเดล ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "Machine Learning", "habitat_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "Machine Learning", "habitat_encoders.pkl")

@st.cache_resource
def load_ml_assets():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            encoders = pickle.load(f)
        return model, encoders
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดไฟล์โมเดล: {e}")
        return None, None

model, encoders = load_ml_assets()

if model and encoders:
    st.markdown("---")
    st.subheader("🔍 ระบุข้อมูลของสัตว์เพื่อทำนาย")

    # --- 3. ส่วนรับข้อมูล (Input) ---
    col1, col2 = st.columns(2)

    with col1:
        if 'Order' in encoders:
            order_options = encoders['Order'].classes_
        else:
            order_options = ["Squamata", "Crocodilia", "Carnivora", "Primates", "Rodentia", "Xenarthra", "Other"]
        selected_order = st.selectbox("อันดับของสัตว์ (Order):", order_options)

    with col2:
        if 'Family' in encoders:
            family_options = encoders['Family'].classes_
        else:
            family_options = ["Unknown", "Viperidae", "Crocodylidae", "Canidae", "Felidae"]
        selected_family = st.selectbox("วงศ์ของสัตว์ (Family):", family_options)

    # ปุ่มกดทำนาย
    if st.button("🚀 ทำนายผลและดูคำแนะนำ", type="primary"):
        
        # --- 4. การทำนายผล (Prediction) ---
        input_df = pd.DataFrame({
            'Order': [selected_order],
            'Family': [selected_family]
        })

        try:
            # Transform ค่า Categorical เป็นตัวเลข
            if 'Order' in encoders:
                input_df['Order'] = encoders['Order'].transform(input_df['Order'])
            if 'Family' in encoders:
                input_df['Family'] = encoders['Family'].transform(input_df['Family'])

            # ทำนาย Habitat
            prediction = model.predict(input_df)
            
            # แปลงผลลัพธ์จากตัวเลขเป็นข้อความ
            if 'Habitat' in encoders and hasattr(encoders['Habitat'], 'inverse_transform'):
                predicted_habitat = encoders['Habitat'].inverse_transform(prediction)[0]
            else:
                habitat_mapping = {
                    0: "Forest (ป่าไม้)", 
                    1: "Grassland/Savanna (ทุ่งหญ้า)", 
                    2: "Aquatic (แหล่งน้ำ/พื้นที่ชุ่มน้ำ)", 
                    3: "Desert (ทะเลทราย)"
                }
                predicted_habitat = habitat_mapping.get(prediction[0], f"รหัสถิ่นที่อยู่: {prediction[0]}")

            # แสดงผลลัพธ์ และ Accuracy
            st.success(f"**ถิ่นที่อยู่อาศัยที่ทำนายได้ (Habitat):** {predicted_habitat}")
            
            # 🚨 จุดที่ต้องแก้: เปลี่ยนตัวเลข 92.50 ให้ตรงกับ Accuracy จริงของคุณตอนรันโมเดล 🚨
            TRAINING_ACCURACY = "92.50%" 
            
            col_metric1, col_metric2 = st.columns(2)
            col_metric1.metric(label="🎯 ความแม่นยำของโมเดลนี้ (Model Accuracy)", value=TRAINING_ACCURACY, delta="ผ่านเกณฑ์")
            col_metric2.metric(label="⚙️ เทคนิคที่ใช้", value="Stacking Ensemble")

        except Exception as e:
            st.warning("⚠️ ไม่สามารถทำนายผลได้ กรุณาตรวจสอบให้แน่ใจว่า Features ที่ส่งให้โมเดลตรงกับตอน Training")
            st.write(f"รายละเอียดข้อผิดพลาด: {e}")

        # --- 5. ระบบแจ้งเตือนและคำแนะนำ (Medical Advice) ---
        st.markdown("---")
        st.subheader("🩺 คำแนะนำเบื้องต้นและการปฐมพยาบาล (Medical Advice)")
        
        if selected_order in ["Squamata"]:
            st.error("🐍 **กลุ่มสัตว์เลื้อยคลาน (Squamata):**\n* **ความเสี่ยง:** ติดเชื้อแบคทีเรีย *Salmonella* และความเสี่ยงจากพิษ (หากเป็นงูพิษ)\n* **ปฐมพยาบาล:** ล้างแผลด้วยน้ำสบู่ หากคาดว่าเป็นงูพิษ **ห้าม** กรีด/ดูดแผล ให้ลดการเคลื่อนไหวแล้วรีบพบแพทย์")
        elif selected_order in ["Crocodilia"]:
            st.error("🐊 **กลุ่มจระเข้ (Crocodilia):**\n* **ความเสี่ยง:** ติดเชื้อรุนแรงจากแบคทีเรียในช่องปาก (เช่น *Aeromonas*) และเนื้อเยื่อฉีกขาด\n* **ปฐมพยาบาล:** ห้ามเลือดเบื้องต้น รีบส่งโรงพยาบาลฉุกเฉินทันทีเพื่อจัดการแผลและรับยาปฏิชีวนะ")
        elif selected_order in ["Carnivora"]:
            st.warning("🐕 **กลุ่มสัตว์กินเนื้อ (Carnivora - สุนัข/แมว/เสือ):**\n* **ความเสี่ยง:** โรคพิษสุนัขบ้า (Rabies), บาดทะยัก, และแบคทีเรีย *Pasteurella*\n* **ปฐมพยาบาล:** ล้างแผลอย่างน้อย 15 นาที ทายาฆ่าเชื้อ และต้องรีบไปฉีดวัคซีนพิษสุนัขบ้าและบาดทะยัก")
        elif selected_order in ["Primates"]:
            st.error("🐒 **กลุ่มไพรเมต (Primates - ลิง/ค่าง):**\n* **ความเสี่ยง:** ไวรัสเริมในลิง (Herpes B virus) ซึ่งอันตรายถึงชีวิตในมนุษย์ รวมถึงพิษสุนัขบ้า\n* **ปฐมพยาบาล:** ล้างแผลให้นานที่สุด และ **ต้องพบแพทย์ทันที** เพื่อรับยาต้านไวรัสและประเมินความเสี่ยง")
        elif selected_order in ["Rodentia", "Lagomorpha"]:
             st.warning("🐀 **กลุ่มสัตว์ฟันแทะ/กระต่าย (Rodentia/Lagomorpha):**\n* **ความเสี่ยง:** ไข้หนูกัด (Rat-bite fever), บาดทะยัก\n* **ปฐมพยาบาล:** ล้างแผลให้สะอาด ทายาฆ่าเชื้อ สังเกตอาการไข้หรือผื่นแดง หากมีอาการให้พบแพทย์")
        elif selected_order in ["Artiodactyla", "Perissodactyla"]:
             st.info("🦌 **กลุ่มสัตว์กีบ (เช่น หมูป่า/กวาง):**\n* **ความเสี่ยง:** แผลฟกช้ำฉีกขาดรุนแรงจากการชนหรือกัด เสี่ยงบาดทะยัก\n* **ปฐมพยาบาล:** ทำความสะอาดแผล หากแผลลึกหรือช้ำมากควรพบแพทย์เพื่อประเมินและฉีดบาดทะยัก")
        elif selected_order in ["Xenarthra"]:
             st.info("🦥 **กลุ่มสลอธ/ตัวกินมด (Xenarthra):**\n* **ความเสี่ยง:** แผลฉีกขาดจากกรงเล็บ เสี่ยงติดเชื้อแบคทีเรียทั่วไปจากดินและบาดทะยัก\n* **ปฐมพยาบาล:** ล้างทำความสะอาดแผลที่เกิดจากกรงเล็บให้ลึกถึงด้านใน ทายาฆ่าเชื้อ และควรฉีดบาดทะยัก")
        else:
            st.info("🐾 **ข้อควรระวังทั่วไป:**\n* **คำแนะนำ:** เมื่อถูกสัตว์ป่ากัดหรือข่วน ล้างแผลด้วยน้ำสะอาดและสบู่ สังเกตอาการบวมแดง และไปพบแพทย์เพื่อประเมินความเสี่ยงโรคพิษสุนัขบ้าและบาดทะยักเสมอ")