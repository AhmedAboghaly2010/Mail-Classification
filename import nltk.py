
import nltk
import os

# تحديد مجلد لحفظ بيانات nltk
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# إضافة المسار إلى قائمة مسارات nltk
nltk.data.path.append(nltk_data_path)

# تنزيل punkt إذا لم يكن موجودًا
nltk.download('punkt', download_dir=nltk_data_path)

