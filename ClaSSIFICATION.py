
import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')

# تحميل النموذج والـ Vectorizer
import os

# تحميل النموذج
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# تحميل الـ vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

st.title("Email Classifier (Spam/Ham)")

def preprocess_text(text):
    ps = PorterStemmer()
    text = text.lower()  # تحويل إلى حروف صغيرة
    text = nltk.word_tokenize(text)  # تقسيم إلى كلمات
    text = [word for word in text if word.isalnum()]  # إزالة الرموز
    text = [word for word in text if word not in stopwords.words('english')]  # إزالة الكلمات الشائعة
    text = [ps.stem(word) for word in text]  # تقليل الكلمات إلى الجذر
    return " ".join(text)


# إدخال المستخدم
input_sms = st.text_input("Enter Your Message")
Bu=st.button("Predict")

if input_sms and Bu:
    # معالجة النص
    processed_sms = preprocess_text(input_sms)

    # تحويل النص باستخدام TF-IDF
    vector_input = vectorizer.transform([processed_sms])

    # التنبؤ باستخدام النموذج
    result = model.predict(vector_input)[0]

    # عرض النتيجة
    if result == 1:
        #st.header("Spam")
        st.image('https://emailchef.com/wp-content/uploads/2019/06/email-spam-reputation.png',width=200)
    else:
       # st.header("Ham")
        st.image('https://miro.medium.com/v2/resize:fit:1100/format:webp/1*z9JN5ps8lpBB9LNLJB3iAg.png',width=300)

