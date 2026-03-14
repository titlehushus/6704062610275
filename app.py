import streamlit as st

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Wildlife & Agriculture AI Project", page_icon="🐾", layout="wide")

# 2. ฟังก์ชันสำหรับหน้า Project Info
def project_info_page():
    st.title("📂 ข้อมูลรายละเอียดโปรเจค (Project Documentation)")
    st.markdown("""
    ยินดีต้อนรับสู่ระบบพยากรณ์อัจฉริยะ โครงการนี้ถูกพัฒนาขึ้นเพื่อประยุกต์ใช้เทคนิคทาง **Machine Learning** และ **Deep Learning** 
    ในการแก้ปัญหาการจำแนกประเภทข้อมูลที่มีความซับซ้อน โดยแบ่งออกเป็น 2 โมเดลหลัก ซึ่งจัดการกับข้อมูลที่แตกต่างกัน 
    ทั้งข้อมูลเชิงโครงสร้าง (Structured Data) และข้อมูลภาพ (Unstructured Data)
    """)
    st.markdown("---")

    tab1, tab2 = st.tabs([
        "🌳 โมเดลที่ 1: Ensemble Learning (Habitat Prediction)",
        "🧠 โมเดลที่ 2: Neural Network (Image Recognition)"
    ])

    # ===================== TAB 1: ENSEMBLE =====================
    with tab1:
        st.header("โมเดลที่ 1: Machine Learning แบบ Ensemble")
        st.write("โมเดลนี้ใช้ข้อมูลทางอนุกรมวิธาน (Taxonomy) เพื่อทำนายถิ่นที่อยู่อาศัยของสัตว์ป่า พร้อมให้คำแนะนำเบื้องต้นทางการแพทย์เมื่อถูกสัตว์กัด")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 ที่มาและรายละเอียดของ Dataset")
            st.markdown("""
            - **ชุดข้อมูล:** Animal Kingdom Taxonomy ดาวน์โหลดจาก [Kaggle](https://www.kaggle.com/datasets/hasanarcas/animal-kingdom-taxonomy)
            - **ขนาดข้อมูล:** ครอบคลุมสัตว์หลายพันสปีชีส์ จำแนกตามลำดับชั้นทางชีววิทยา
            - **Features ที่ใช้เทรน:** `Order` (อันดับ) และ `Family` (วงศ์) ของสัตว์
            - **Target (เป้าหมาย):** ถิ่นที่อยู่อาศัย (Habitat) ได้แก่ Forest/Urban, Grassland, Ground/Trees, General Terrestrial
            """)

            st.subheader("🛠️ กระบวนการเตรียมข้อมูล (Data Preprocessing)")
            st.markdown("""
            ข้อมูลดิบจาก Kaggle มีความไม่สมบูรณ์หลายจุด จึงต้องผ่านขั้นตอนดังนี้:

            **1. จัดการค่าว่าง (Handling Missing Values)**
            - ตรวจสอบด้วย `df.dropna()` บน columns `Order` และ `Family`
            - ตัดแถวที่ไม่มีข้อมูล Order หรือ Family ออก เพื่อให้ข้อมูลที่เหลือสมบูรณ์ 100%

            **2. แปลงข้อมูลข้อความเป็นตัวเลข (Label Encoding)**
            - ชื่อสัตว์และวงศ์เป็น Text ซึ่งโมเดลไม่สามารถรับได้โดยตรง
            - ใช้ `LabelEncoder` จาก scikit-learn แปลงทั้ง `Order`, `Family` และ `Target_Habitat`
            - บันทึก encoder ไว้ใน `habitat_encoders.pkl` เพื่อใช้แปลงข้อมูลใหม่ตอน predict

            **3. สร้าง Target Label (Feature Engineering)**
            - Dataset ต้นฉบับไม่มีคอลัมน์ Habitat โดยตรง จึงสร้างขึ้นจากกฎอนุกรมวิธาน เช่น:
              - `Carnivora` → Forest/Urban
              - `Artiodactyla` → Grassland  
              - `Squamata` → Ground/Trees
              - อื่นๆ → General Terrestrial
            """)

        with col2:
            st.subheader("🔬 โครงสร้างอัลกอริทึม (Voting Ensemble)")
            st.markdown("""
            ใช้เทคนิค **Soft Voting Ensemble** ผสานจุดแข็งของ 2 อัลกอริทึม:

            **Base Model 1: Random Forest**
            - `n_estimators=20` (สร้าง 20 ต้นไม้)
            - `random_state=42`
            - บทบาท: สร้าง Decision Trees หลายต้นพร้อมกัน แล้วโหวตหาผลลัพธ์
            - ข้อดี: ทนทานต่อ Noise ลด Overfitting

            **Base Model 2: Gradient Boosting**
            - `n_estimators=10`
            - `random_state=42`
            - บทบาท: เรียนรู้แบบต่อเนื่อง แต่ละรอบแก้ข้อผิดพลาดของรอบก่อน
            - ข้อดี: จับ Pattern ที่ซับซ้อนได้ดีกว่า Random Forest เดี่ยวๆ

            **การรวมผล (Soft Voting)**
            - รวมค่าความน่าจะเป็น (Probability) จากทั้ง 2 โมเดล
            - คลาสที่ได้ค่าเฉลี่ยความน่าจะเป็นสูงสุดจะเป็นคำตอบสุดท้าย
            """)
            st.info("💡 Soft Voting ให้ผลแม่นยำกว่า Hard Voting เพราะนำ Confidence ของแต่ละโมเดลมาถ่วงน้ำหนักด้วย")

            st.subheader("📊 ผลลัพธ์การเทรน")
            col_a, col_b = st.columns(2)
            col_a.metric("Ensemble Accuracy", "~92%", "Voting: Soft")
            col_b.metric("จำนวน Target Classes", "4 คลาส", "Habitat types")

    # ===================== TAB 2: NEURAL NETWORK =====================
    with tab2:
        st.header("โมเดลที่ 2: Neural Network ด้วย Transfer Learning (MobileNetV2)")
        st.write("ระบบจำแนกประเภทภาพถ่ายผักและผลไม้ 36 ชนิด โดยอาศัย Pretrained Model เป็นฐาน พร้อมต่อยอด Classification Head ใหม่")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 ที่มาและรายละเอียดของ Dataset")
            st.markdown("""
            - **ชุดข้อมูล:** Fruit and Vegetable Image Recognition จาก [Kaggle](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)
            - **จำนวนคลาส:** 36 ชนิด (เช่น apple, banana, carrot, tomato ...)
            - **แบ่งข้อมูล 3 ส่วน:**
              - Train: 3,115 ภาพ
              - Validation: 351 ภาพ
              - Test: 359 ภาพ
            - **ประเภทข้อมูล:** รูปภาพ (Unstructured Data) หลากหลายมุมกล้อง แสง และพื้นหลัง
            """)

            st.subheader("🛠️ การเตรียมข้อมูลภาพ (Image Preprocessing)")
            st.markdown("""
            **1. Resizing**
            - ปรับทุกภาพให้เป็น **224 × 224 pixels** เพื่อให้ตรงกับ Input ของ MobileNetV2

            **2. Pixel Normalization**
            - หารค่าพิกเซลด้วย 255.0 → ค่าอยู่ในช่วง **[0, 1]**
            - ช่วยให้โมเดล Converge เร็วขึ้นและเสถียรกว่า

            **3. Data Augmentation (เฉพาะชุด Train)**
            - `rotation_range=20` — หมุนภาพแบบสุ่ม ±20 องศา
            - `width_shift_range=0.2` — เลื่อนภาพซ้าย/ขวา 20%
            - `height_shift_range=0.2` — เลื่อนภาพขึ้น/ลง 20%
            - `horizontal_flip=True` — พลิกภาพในแนวนอน
            - **ผล:** เพิ่มความหลากหลายของข้อมูล ป้องกัน Overfitting
            - ชุด Validation และ Test **ไม่** ทำ Augmentation เพื่อวัดผลบนภาพจริง
            """)

        with col2:
            st.subheader("🧠 โครงสร้างสถาปัตยกรรม (Transfer Learning)")
            st.markdown("""
            ใช้เทคนิค **Transfer Learning** โดยนำ MobileNetV2 ที่เทรนบน ImageNet มาต่อยอด:

            **ส่วนที่ 1: Base Model (MobileNetV2)**
            - โหลดน้ำหนักจาก ImageNet (`weights='imagenet'`)
            - `include_top=False` — ตัดส่วน Classification Head เดิมออก
            - `trainable=False` — แช่แข็ง (Freeze) น้ำหนักเดิมทั้งหมด
            - ทำหน้าที่สกัด Feature จากรูปภาพแทนการสร้างใหม่จากศูนย์

            **ส่วนที่ 2: Custom Classification Head**
            - `GlobalAveragePooling2D()` — ลด Feature Map ให้เป็น Vector
            - `Dense(256, activation='relu')` — Hidden Layer 256 โหนด
            - `Dropout(0.5)` — สุ่มปิด 50% ของโหนด ป้องกัน Overfitting
            - `Dense(36, activation='softmax')` — Output 36 คลาส

            **การ Compile**
            - Optimizer: `Adam`
            - Loss: `categorical_crossentropy`
            - Metrics: `accuracy`
            """)

            st.subheader("📊 ผลลัพธ์การเทรน (10 Epochs)")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Test Accuracy", "94.99%", "↑ จาก Epoch 1: 40%")
            col_b.metric("Val Accuracy", "95.16%", "Epoch 10")
            col_c.metric("จำนวน Classes", "36 ชนิด")
            
            st.markdown("""
            | Epoch | Train Acc | Val Acc |
            |-------|-----------|---------|
            | 1     | 40.58%    | 81.48%  |
            | 3     | 72.01%    | 90.03%  |
            | 5     | 78.65%    | 91.74%  |
            | 8     | 81.99%    | 93.45%  |
            | 10    | 84.62%    | 95.16%  |
            """)

    # ===================== REFERENCES =====================
    st.markdown("---")
    st.subheader("🔗 แหล่งอ้างอิงและเทคโนโลยีที่ใช้ (References & Tech Stack)")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📦 Datasets**")
        st.markdown("- [Animal Kingdom Taxonomy — Kaggle](https://www.kaggle.com/datasets/hasanarcas/animal-kingdom-taxonomy)")
        st.markdown("- [Fruit and Vegetable Image Recognition — Kaggle](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)")
    with col2:
        st.markdown("**🤖 Machine Learning**")
        st.markdown("- `scikit-learn` — RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, LabelEncoder")
        st.markdown("- `pandas` / `numpy` — Data manipulation")
    with col3:
        st.markdown("**🧠 Deep Learning & UI**")
        st.markdown("- `TensorFlow 2.x` / `Keras` — MobileNetV2, ImageDataGenerator")
        st.markdown("- `Streamlit` — Web application framework")
        st.markdown("- `Pillow` — Image processing")

# 3. Navigation
pg = st.navigation({
    "เมนูหลัก": [
        st.Page(project_info_page, title="ข้อมูลโปรเจค", icon="📂"),
    ],
    "ระบบพยากรณ์": [
        st.Page("pages/1_Ensemble.py", title="Ensemble Test", icon="🌳"),
        st.Page("pages/2_NN.py", title="Neural Network Test", icon="🧠"),
    ]
})

pg.run()