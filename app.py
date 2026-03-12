import streamlit as st

# 1. ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Wildlife & Agriculture AI Project", page_icon="🐾", layout="wide")

# 2. ฟังก์ชันสำหรับหน้า Project Info
def project_info_page():
    st.title("📂 ข้อมูลรายละเอียดโปรเจค (Project Documentation)")
    st.markdown("""
    ยินดีต้อนรับสู่ระบบพยากรณ์อัจฉริยะ โครงการนี้ถูกพัฒนาขึ้นเพื่อประยุกต์ใช้เทคนิคทาง **Machine Learning** และ **Deep Learning** ในการแก้ปัญหาการจำแนกประเภทข้อมูลที่มีความซับซ้อน โดยแบ่งขอบเขตการทำงานออกเป็น 2 โมเดลหลัก ซึ่งจัดการกับข้อมูลที่แตกต่างกัน 
    ทั้งข้อมูลเชิงโครงสร้าง (Structured Data) และข้อมูลภาพ (Unstructured Data)
    """)
    st.markdown("---")
    
    # แบ่งเป็น 2 Tabs สำหรับ 2 โมเดล ตามที่อาจารย์กำหนด 
    tab1, tab2 = st.tabs(["🌳 โมเดลที่ 1: Ensemble Learning (Habitat Prediction)", "🧠 โมเดลที่ 2: Neural Network (Image Recognition)"])

    with tab1:
        st.header("1. Machine Learning แบบ Ensemble")
        st.write("โมเดลนี้มุ่งเน้นการวิเคราะห์ข้อมูลทางอนุกรมวิธาน (Taxonomy) เพื่อทำนายถิ่นที่อยู่อาศัยของสัตว์ป่า รวมถึงให้คำแนะนำเบื้องต้นทางการแพทย์เมื่อเกิดเหตุฉุกเฉิน")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📝 รายละเอียดและที่มาของข้อมูล")
            st.write("- **ชุดข้อมูล (Dataset):** Animal Kingdom Taxonomy จากแหล่งข้อมูล Kaggle")
            st.write("- **คุณลักษณะ (Features):** ใช้ข้อมูลการจัดหมวดหมู่ทางชีววิทยา (เช่น Order, Family) เป็นตัวแปรต้น เพื่อวิเคราะห์ความสัมพันธ์กับสภาพแวดล้อมที่สัตว์อาศัยอยู่")
            st.write("- **ความท้าทายของข้อมูล:** ข้อมูลในธรรมชาติมักมีการกระจายตัวไม่เท่ากัน สัตว์บางกลุ่มมีจำนวนสปีชีส์เยอะมาก ทำให้เกิดปัญหา Class Imbalance")
            
            st.subheader("🛠️ กระบวนการเตรียมข้อมูล (Data Preprocessing)")
            st.markdown("""
            เพื่อให้โมเดลมีประสิทธิภาพสูงสุดและตรงตามมาตรฐาน ได้ผ่านขั้นตอนดังนี้:
            1. **Handling Missing Values:** ตรวจสอบและจัดการค่าว่าง (Null) ด้วยเทคนิค Imputation เพื่อไม่ให้สูญเสียข้อมูลสำคัญ
            2. **Categorical Encoding:** แปลงข้อมูลข้อความ (ชื่อสายพันธุ์/วงศ์) ให้เป็นตัวเลขด้วย Label/One-Hot Encoding
            3. **Feature Scaling:** ปรับสเกลข้อมูลให้อยู่ในมาตรฐานเดียวกันด้วย `StandardScaler`
            4. **Imbalanced Data Handling (SMOTE):** แก้ปัญหาคลาสที่ไม่สมดุลด้วยการสังเคราะห์ข้อมูลเพิ่ม (Synthetic Minority Over-sampling Technique) เพื่อไม่ให้โมเดลเอนเอียงไปยังคลาสที่มีจำนวนมาก
            """)

        with col2:
            st.subheader("🔬 โครงสร้างอัลกอริทึม (Stacking Ensemble)")
            st.markdown("""
            เพื่อเพิ่มขีดความสามารถในการพยากรณ์ โปรเจคนี้เลือกใช้เทคนิค **Stacking Ensemble** โดยผสานจุดแข็งของ 3 อัลกอริทึมเข้าด้วยกัน:
            
            * **Base Learner 1: Random Forest**
                * *บทบาท:* สร้าง Decision Trees ย่อยๆ จำนวนมาก และโหวตหาผลลัพธ์
                * *ข้อดี:* ทนทานต่อ Noise และลดความเสี่ยงในการเกิด Overfitting
            * **Base Learner 2: XGBoost (Extreme Gradient Boosting)**
                * *บทบาท:* เรียนรู้และแก้ไขข้อผิดพลาด (Residuals) ของโมเดลก่อนหน้าอย่างต่อเนื่อง
                * *ข้อดี:* มีประสิทธิภาพสูง จับความสัมพันธ์ที่ซับซ้อน (Non-linear) ได้ดีเยี่ยม
            * **Meta-Learner: Logistic Regression**
                * *บทบาท:* นำผลลัพธ์ความน่าจะเป็นจาก Random Forest และ XGBoost มาเป็น Input เพื่อตัดสินใจหาคำตอบสุดท้าย
                * *ข้อดี:* ช่วยถ่วงน้ำหนักว่าควรเชื่อโมเดลพื้นฐานตัวไหนในบริบทใด ทำให้ผลลัพธ์สุดท้ายเสถียรที่สุด
            """)
            st.info("💡 กระบวนการนี้ทำให้ได้ค่า Accuracy ที่สูงกว่าการใช้โมเดลเดี่ยวๆ อย่างมีนัยสำคัญ")

    with tab2:
        st.header("2. Neural Network (Artificial Neural Network)")
        st.write("ระบบวิเคราะห์และจำแนกประเภทภาพถ่ายผักและผลไม้โดยอัตโนมัติ เพื่อประยุกต์ใช้ในอุตสาหกรรมการเกษตรและการคัดแยกสินค้า")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📝 รายละเอียดและที่มาของข้อมูล")
            st.write("- **ชุดข้อมูล (Dataset):** Fruit and Vegetable Image Recognition จาก Kaggle")
            st.write("- **ประเภทข้อมูล:** รูปภาพ (Unstructured Data) ที่มีความหลากหลายทั้งด้านแสง เงา และมุมกล้อง")
            
            st.subheader("🛠️ การจัดการข้อมูลภาพ (Image Preprocessing)")
            st.markdown("""
            การจัดการรูปภาพก่อนเข้าสู่ Neural Network ถือเป็นหัวใจสำคัญ:
            1. **Image Resizing:** ปรับสัดส่วนรูปภาพทั้งหมดให้มีขนาดเท่ากัน (เช่น 224x224 พิกเซล) เพื่อให้เข้ากับ Input Layer
            2. **Pixel Normalization:** ปรับค่าพิกเซลจาก 0-255 ให้อยู่ในช่วง 0-1 (แบ่งด้วย 255.0) เพื่อให้โมเดลคอนเวอร์จ (Converge) เข้าสู่จุดต่ำสุดได้เร็วขึ้น
            3. **Data Augmentation:** สุ่มหมุนภาพ พลิกภาพ หรือปรับแสงระหว่างการเทรน เพื่อเพิ่มความหลากหลาย และป้องกันปัญหาโมเดลจำข้อสอบ (Overfitting)
            """)

        with col2:
            st.subheader("🧠 โครงสร้างสถาปัตยกรรม (Custom ANN Architecture)")
            st.markdown("""
            โมเดลนี้ได้รับการออกแบบโครงสร้าง (Network Topology) ขึ้นมาเองทั้งหมด (From Scratch) โดยไม่ใช้ Pre-trained Model:
            
            * **Input Layer:** รับเมทริกซ์รูปภาพและทำการ `Flatten` ให้เป็นเวกเตอร์ 1 มิติ
            * **Hidden Layers (Multi-Layer Perceptron):**
                * ประกอบด้วยชั้น Dense Layer ไม่ต่ำกว่า 3 ชั้น
                * ใช้ **ReLU (Rectified Linear Unit)** เป็น Activation Function เพื่อแก้ปัญหา Vanishing Gradient
                * มีการแทรก **Dropout Layer** (เช่น 0.2 - 0.5) เพื่อสุ่มปิดการทำงานของบางโหนด บังคับให้เน็ตเวิร์กกระจายการเรียนรู้
            * **Output Layer:**
                * จำนวนโหนดเท่ากับจำนวนคลาสของผักและผลไม้ทั้งหมด
                * ใช้ **Softmax Function** แปลงค่าลอจิต (Logits) ให้ออกมาเป็นค่าความน่าจะเป็น (Probability) รวมกันได้ 1.0 พอดี
            """)
            st.success("🎯 การออกแบบโครงสร้างนี้ ผ่านกระบวนการทดลองปรับจูน Hyperparameters เพื่อให้สอดคล้องกับขนาดของ Dataset มากที่สุด")

    st.markdown("---")
    st.subheader("🔗 แหล่งอ้างอิงและเทคโนโลยี (References & Tech Stack)")
    st.write("- **Datasets:** [Animal Kingdom Taxonomy](https://www.kaggle.com/datasets/hasanarcas/animal-kingdom-taxonomy) | [Fruit and Vegetable Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)")
    st.write("- **Core Frameworks:** `Python`, `Streamlit` (UI), `Pandas`, `NumPy`")
    st.write("- **Machine Learning:** `Scikit-learn` (Pipeline, Encoders), `XGBoost`")
    st.write("- **Deep Learning:** `TensorFlow` / `Keras`")

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