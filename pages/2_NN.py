import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import tensorflow as tf

# 🚨 HOTFIX: แก้อาการไลบรารีตีกัน (Monkey Patch) 🚨
# โค้ดนี้ทำหน้าที่อุดรอยรั่วของเวอร์ชัน เพื่อหลอกให้ tf_keras ทำงานได้ โดยไม่ต้องไปแก้ requirements ให้เว็บพังอีก
try:
    def _dummy_register(*args, **kwargs): 
        pass
    
    if not hasattr(tf, '__internal__'):
        class _Internal: pass
        tf.__internal__ = _Internal()
        
    tf.__internal__.register_load_context_function = _dummy_register
    
    import tensorflow._api.v2.compat.v2.internal as _tf_internal
    _tf_internal.register_load_context_function = _dummy_register
except Exception:
    pass
# --------------------------------------------------------

import tf_keras
from PIL import Image
import numpy as np
import json

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Neural Network", layout="wide", page_icon="🧠")

# --- 2. การจัดการ Path และโหลดโมเดล ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_nn_model():
    nn_path = os.path.join(BASE_DIR, 'Neural Network', 'nn_model.h5')
    lbl_path = os.path.join(BASE_DIR, 'Neural Network', 'class_labels.json')
    
    try:
        if os.path.exists(nn_path):
            # โหลดด้วย tf_keras พร้อมปิด compile เพื่อเลี่ยง Error โครงสร้าง
            model = tf_keras.models.load_model(nn_path, compile=False)
            labels = None
            
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

# 🚨 เปลี่ยนตัวเลขให้ตรงกับ Accuracy จริงของคุณตอนเทรน 
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
                # 4. Preprocessing
                img = image.convert('RGB').resize((224, 224))
                img_array = np.array(img) / 255.0  
                img_array = np.expand_dims(img_array, axis=0) 
                
                # 5. Prediction
                preds = nn_model.predict(img_array)
                idx = np.argmax(preds)
                idx_str = str(idx)
                
                if labels:
                    result = labels.get(idx_str, f"คลาส {idx_str}")
                else:
                    result = f"คลาส {idx_str}"
                
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
    target_path = os.path.join(BASE_DIR, 'Neural Network', 'nn_model.h5')
    st.error(f"⚠️ ไม่พบไฟล์โมเดลที่ตำแหน่ง: {target_path}")
    st.info("กรุณาตรวจสอบว่าชื่อโฟลเดอร์ 'Neural Network' และไฟล์ 'nn_model.h5' สะกดถูกต้อง")