import os
import streamlit as st
import numpy as np
from PIL import Image
import json
import tensorflow as tf
from tensorflow import keras

# --- Patch: รองรับ Keras เวอร์ชันใหม่ที่มี quantization_config ---
class _PatchedDense(keras.layers.Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

class _PatchedConv2D(keras.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

CUSTOM_OBJECTS = {
    'Dense': _PatchedDense,
    'Conv2D': _PatchedConv2D,
}

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Neural Network", layout="wide", page_icon="🧠")

# --- 2. โหลดโมเดล ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_nn_model():
    nn_path = os.path.join(BASE_DIR, 'Neural Network', 'nn_model.h5')
    lbl_path = os.path.join(BASE_DIR, 'Neural Network', 'class_labels.json')

    if not os.path.exists(nn_path):
        return None, None, nn_path

    try:
        model = tf.keras.models.load_model(
            nn_path,
            compile=False,
            custom_objects=CUSTOM_OBJECTS
        )
        labels = None
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r', encoding='utf-8') as f:
                labels = json.load(f)
        return model, labels, None

    except Exception as e:
        return None, None, str(e)

# --- 3. หน้าเว็บ ---
st.title("🧠 ทดสอบระบบจำแนกภาพ (Neural Network)")
st.write("อัปโหลดรูปภาพผักหรือผลไม้ เพื่อให้โมเดล AI ทำการจำแนกประเภท")

nn_model, labels, error_msg = load_nn_model()

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
                img = image.convert('RGB').resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                preds = nn_model.predict(img_array)
                idx = int(np.argmax(preds))
                idx_str = str(idx)

                result = labels.get(idx_str, f"คลาส {idx_str}") if labels else f"คลาส {idx_str}"
                confidence = float(np.max(preds)) * 100

                st.markdown("### ผลลัพธ์การวิเคราะห์")
                st.success(f"🎉 ตรวจพบว่าเป็น: **{result}**")

                col_m1, col_m2 = st.columns(2)
                col_m1.metric(label="ความมั่นใจ (Confidence)", value=f"{confidence:.2f}%")
                col_m2.metric(label="ความแม่นยำรวม (Overall Accuracy)", value=TRAINING_ACCURACY)

                st.write("ระดับความมั่นใจในภาพนี้:")
                st.progress(int(confidence))
else:
    if error_msg and '/' not in str(error_msg):
        st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {error_msg}")
        st.info("💡 วิธีแก้: บันทึกโมเดลใหม่ด้วย model.save() บน Keras เวอร์ชันเดียวกับ server")
    else:
        target_path = os.path.join(BASE_DIR, 'Neural Network', 'nn_model.h5')
        st.error(f"⚠️ ไม่พบไฟล์โมเดลที่ตำแหน่ง: {target_path}")
        st.info("กรุณาตรวจสอบว่าชื่อโฟลเดอร์ 'Neural Network' และไฟล์ 'nn_model.h5' สะกดถูกต้อง")