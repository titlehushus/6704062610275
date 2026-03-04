import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

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
nn_model, labels = load_nn_model()

if nn_model:
    uploaded_file = st.file_uploader("อัปโหลดรูปภาพ", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="รูปที่อัปโหลด", width=400)
        
        with st.spinner('กำลังประมวลผล...'):
            img = image.convert('RGB').resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            preds = nn_model.predict(img_array)
            idx = str(np.argmax(preds))
            result = labels.get(idx, f"คลาส {idx}") if labels else f"คลาส {idx}"
            
            st.success(f"🎉 ผลการทำนาย: **{result}**")
else:
    st.error("⚠️ ไม่พบไฟล์โมเดลในโฟลเดอร์ Neural Network")