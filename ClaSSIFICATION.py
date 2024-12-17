import nltk
import os

# تحديد مسار محلي لمجلد nltk_data
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# إضافة المسار إلى قائمة مسارات NLTK
nltk.data.path.append(nltk_data_path)

# تحميل punkt إلى هذا المسار
nltk.download('punkt', download_dir=nltk_data_path)

import streamlit as st
import pickle
import nltk
nltk.download('punkt')
#nltk.download()
#from nltk.corpus import brown
#brown.words()
from nltk.corpus import stopwords , brown
from nltk.stem import PorterStemmer
# تحميل بيانات NLTK
nltk.download('stopwords')


# تحميل النموذج والـ vectorizer
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError:
    st.error("Model or Vectorizer file not found. Please check file paths.")

st.title("Email Classifier (Spam/Ham)")

# دالة المعالجة
def preprocess_text(text):
    try:
        ps = PorterStemmer()
        text = text.lower()  # تحويل إلى حروف صغيرة
        text = nltk.word_tokenize(text)  # تقسيم النص إلى كلمات
        text = [word for word in text if word.isalnum()]  # إزالة الرموز
        text = [word for word in text if word not in stopwords.words('english')]  # إزالة الكلمات الشائعة
        text = [ps.stem(word) for word in text]  # تقليل الكلمات إلى الجذر
        return " ".join(text)
    except Exception as s:
        st.error(f"Error occurred during preprocessing: {s}")
        return ""

# إدخال المستخدم
input_sms = st.text_input("Enter Your Message")
predict_button = st.button("Predict")

if predict_button and input_sms.strip():
    # معالجة النص
    processed_sms = preprocess_text(input_sms)
    
    if processed_sms:  # إذا كانت المعالجة ناجحة
        try:
            # تحويل النص باستخدام vectorizer
            vector_input = vectorizer.transform([processed_sms])

            # التنبؤ باستخدام النموذج
            result = model.predict(vector_input)[0]

            # عرض النتيجة
            if result == 1:
                st.image('https://emailchef.com/wp-content/uploads/2019/06/email-spam-reputation.png', width=200)
            else:
                st.image('https://miro.medium.com/v2/resize:fit:1100/format:webp/1*z9JN5ps8lpBB9LNLJB3iAg.png', width=300)
        except Exception as e:
            st.error(f"Error occurred during prediction: {e}")
else:
    if predict_button:
        st.warning("Please enter a valid message before predicting.")
