import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Neural Network", layout="wide", page_icon="🧠")

# --- 2. การจัดการ Path และโหลดโมเดล ---
# ถอยออกจากโฟลเดอร์ pages 1 ชั้น เพื่อกลับไปที่ Root (MY_PJ)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_nn_model():
    # ระบุ Path ไปยังโฟลเดอร์ Neural Network โดยตรง
    nn_path = os.path.join(BASE_DIR, 'Neural Network', 'nn_model.h5')
    lbl_path = os.path.join(BASE_DIR, 'Neural Network', 'class_labels.json')
    
    try:
        if os.path.exists(nn_path):
            # โหลดโมเดล .h5
            model = tf.keras.models.load_model(nn_path, compile=False)
            labels = None
            
            # โหลดไฟล์ Label ถ้ามี
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r', encoding='utf-8') as f:
                    labels = json.load(f)
            return model, labels
        else:
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# --- 3. ส่วนแสดงผลหน้าเว็บ ---
st.title("🧠 ทดสอบระบบจำแนกภาพ (Neural Network)")
st.write("อัปโหลดรูปภาพผักหรือผลไม้ เพื่อให้โมเดล AI ทำการจำแนกประเภท")

nn_model, labels = load_nn_model()

# 🚨 จุดที่ต้องแก้: เปลี่ยนตัวเลขให้ตรงกับ Accuracy จริงของคุณตอนเทรน 🚨
TRAINING_ACCURACY = "88.75%"

if nn_model:
    uploaded_file = st.file_uploader("อัปโหลดรูปภาพ", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="รูปที่อัปโหลด", use_container_width=True)
        
        with col2:
            with st.spinner('กำลังให้ AI ประมวลผล...'):
                # 4. Preprocessing (เตรียมรูปภาพ)
                # ปรับขนาดเป็น 224x224 ตามที่โมเดลส่วนใหญ่ต้องการ (หรือปรับตามที่คุณเทรนมา)
                img = image.convert('RGB').resize((224, 224))
                img_array = np.array(img) / 255.0  # Normalization
                img_array = np.expand_dims(img_array, axis=0) # เพิ่มมิติสำหรับ Batch size
                
                # 5. Prediction (การพยากรณ์)
                preds = nn_model.predict(img_array)
                idx = np.argmax(preds)
                idx_str = str(idx)
                
                # ดึงชื่อคลาสจาก JSON labels
                if labels:
                    result = labels.get(idx_str, f"คลาส {idx_str}")
                else:
                    result = f"คลาส {idx_str}"
                
                # คำนวณความมั่นใจ (Confidence Score)
                confidence = float(np.max(preds)) * 100
                
                # 6. แสดงผลลัพธ์
                st.markdown("### ผลลัพธ์การวิเคราะห์")
                st.success(f"🎉 ตรวจพบว่าเป็น: **{result}**")
                
                col_m1, col_m2 = st.columns(2)
                col_m1.metric(label="ความมั่นใจ (Confidence)", value=f"{confidence:.2f}%")
                col_m2.metric(label="ความแม่นยำรวม (Overall Accuracy)", value=TRAINING_ACCURACY)
                
                st.write("ระดับความมั่นใจในภาพนี้:")
                st.progress(int(confidence))
                
else:
    # แสดง Error พร้อมบอก Path ที่ระบบพยายามหาไฟล์
    target_path = os.path.join(BASE_DIR, 'Neural Network', 'nn_model.h5')
    st.error(f"⚠️ ไม่พบไฟล์โมเดลที่ตำแหน่ง: {target_path}")
    st.info("กรุณาตรวจสอบว่าชื่อโฟลเดอร์ 'Neural Network' และไฟล์ 'nn_model.h5' สะกดถูกต้อง (ตัวเล็ก-ใหญ่มีผล)")