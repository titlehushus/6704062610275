import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # สั่งให้ใช้ระบบโหลดโมเดลแบบเก่า

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json

# หา Path ของโฟลเดอร์ Root
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
nn_model_path = os.path.join(root_dir, 'nn_model.h5')
json_path = os.path.join(root_dir, 'class_labels.json')

# โหลด Model และ Class Labels
try:
    model = tf.keras.models.load_model(nn_model_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        class_names = json.load(f)
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")

st.set_page_config(page_title="Neural Network", layout="wide")

@st.cache_resource
def load_nn_model():
    nn_folders = ['Neural Network', 'neural network']
    for folder in nn_folders:
        nn_path = os.path.join(folder, 'nn_model.h5')
        lbl_path = os.path.join(folder, 'class_labels.json')
        if os.path.exists(nn_path):
            model = tf.keras.models.load_model(nn_path)
            labels = None
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r', encoding='utf-8') as f:
                    labels = json.load(f)
            return model, labels
    return None, None

st.title("🧠 ทดสอบระบบจำแนกภาพ (Neural Network)")
st.write("อัปโหลดรูปภาพผักหรือผลไม้ เพื่อให้โมเดล AI ทำการจำแนกประเภท")

nn_model, labels = load_nn_model()

# 🚨 จุดที่ต้องแก้: เปลี่ยนตัวเลข 88.75 ให้ตรงกับ Accuracy จริงของคุณตอนเทรน Neural Network 🚨
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
                # Preprocessing
                img = image.convert('RGB').resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Prediction
                preds = nn_model.predict(img_array)
                idx = np.argmax(preds)
                idx_str = str(idx)
                
                result = labels.get(idx_str, f"คลาส {idx_str}") if labels else f"คลาส {idx_str}"
                
                # คำนวณความมั่นใจ (Confidence Score) ของคลาสที่เลือก
                confidence = float(np.max(preds)) * 100
                
                # แสดงผลลัพธ์
                st.markdown("### ผลลัพธ์การวิเคราะห์")
                st.success(f"🎉 ตรวจพบว่าเป็น: **{result}**")
                
                # แสดงค่า Accuracy เชิงสถิติ
                st.metric(label="เปอร์เซ็นต์ความมั่นใจ (Confidence)", value=f"{confidence:.2f}%", help="ความน่าจะเป็นที่โมเดลคิดว่ารูปนี้ตรงกับผลลัพธ์ที่สุด")
                st.metric(label="ความแม่นยำภาพรวมของโมเดล (Overall Accuracy)", value=TRAINING_ACCURACY, help="อ้างอิงจากคะแนน Accuracy ของ Validation Set ระหว่างการเทรน")
                
                # แถบแสดงความมั่นใจ (Progress Bar)
                st.write("ระดับความมั่นใจ:")
                st.progress(int(confidence))
                
else:
    st.error("⚠️ ไม่พบไฟล์โมเดล (`nn_model.h5`) ในโฟลเดอร์ Neural Network")