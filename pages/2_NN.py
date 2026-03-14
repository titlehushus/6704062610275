# -*- coding: utf-8 -*-
import os
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import json

# 1. ตั้งค่าหน้าเว็บ (ควรทำเป็นอันดับแรก)
st.set_page_config(page_title="Neural Network Test", layout="wide")

# 2. ฟังก์ชันโหลดโมเดลและ Labels (ใช้แคชเพื่อไม่ให้โหลดใหม่ทุกครั้งที่กดปุ่ม)
@st.cache_resource
def load_nn_assets():
    # กำหนดโฟลเดอร์ที่อาจเป็นไปได้ (ตรวจสอบตัวพิมพ์เล็ก-ใหญ่ให้ตรงกับบน GitHub)
    possible_folders = ['Neural Network', 'neural network', '.']
    
    for folder in possible_folders:
        model_path = os.path.join(folder, 'nn_model.h5')
        label_path = os.path.join(folder, 'class_labels.json')
        
        if os.path.exists(model_path):
            try:
                # โหลดโมเดล
                model = tf.keras.models.load_model(model_path)
                # โหลด Labels
                labels = None
                if os.path.exists(label_path):
                    with open(label_path, 'r', encoding='utf-8') as f:
                        labels = json.load(f)
                return model, labels
            except Exception as e:
                st.error(f"พบโมเดลใน {folder} แต่โหลดไม่ได้: {e}")
    return None, None

# 3. เริ่มส่วนการแสดงผล
st.title("🧠 ทดสอบระบบจำแนกภาพ (Neural Network)")
st.write("อัปโหลดรูปภาพผักหรือผลไม้ เพื่อให้ AI ทำการจำแนกประเภท")

# เรียกใช้งานฟังก์ชันโหลดโมเดล
nn_model, labels = load_nn_assets()

# ค่า Accuracy อ้างอิงจากโปรเจค
TRAINING_ACCURACY = "88.75%"

if nn_model:
    uploaded_file = st.file_uploader("อัปโหลดรูปภาพ (JPG, PNG)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        
        # แสดงรูปภาพที่อัปโหลด
        with col1:
            input_img = Image.open(uploaded_file)
            st.image(input_img, caption="รูปภาพที่อัปโหลด", use_container_width=True)
        
        # ประมวลผลและทำนาย
        with col2:
            with st.spinner('AI กำลังคิด...'):
                try:
                    # 1. Preprocessing: ปรับขนาดเป็น 224x224 และทำ Normalization
                    img_resized = input_img.convert('RGB').resize((224, 224))
                    img_array = np.array(img_resized) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # 2. Prediction
                    predictions = nn_model.predict(img_array)
                    class_idx = np.argmax(predictions)
                    confidence = float(np.max(predictions)) * 100
                    
                    # 3. ดึงชื่อคลาสจาก JSON
                    result_text = labels.get(str(class_idx), f"Class {class_idx}") if labels else f"Class {class_idx}"
                    
                    # 4. แสดงผลลัพธ์
                    st.markdown("### 📊 ผลการวิเคราะห์")
                    st.success(f"ตรวจพบว่าเป็น: **{result_text}**")
                    
                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric("ความมั่นใจ (Confidence)", f"{confidence:.2f}%")
                    m_col2.metric("ความแม่นยำรวม (Overall Acc)", TRAINING_ACCURACY)
                    
                    st.progress(int(confidence))
                    
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดระหว่างประมวลผล: {e}")
else:
    st.error("⚠️ ไม่พบไฟล์ `nn_model.h5` ในโฟลเดอร์โครงการ กรุณาตรวจสอบการจัดวางไฟล์")