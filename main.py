import streamlit as st
import easyocr
import numpy as np
from PIL import Image
from deep_translator import GoogleTranslator

# --- НАСТРОЙКИ ---
st.set_page_config(page_title="Pro OCR УрФУ", page_icon="🎯")

@st.cache_resource
def load_reader():
    # Загружаем модель для английского и французского языков
    # Она скачается один раз при первом запуске (около 100-200 МБ)
    return easyocr.Reader(['en', 'fr'])

# --- SIDEBAR (Требование 4 недели) ---
with st.sidebar:
    st.title("О проекте")
    st.markdown("""
    **Программа:** [Цифровые кафедры УрФУ](https://dpo.urfu.ru/programs/92)
    **Команда:** [Твое Имя и Фамилия]
    **Дисциплина:** Возможности Python
    """)
    st.info("Используется модель EasyOCR (Deep Learning)")

# --- ИНТЕРФЕЙС ---
st.title("🎯 Профессиональный распознаватель текста")
st.write("Загрузите фото любого формата — модель сама найдет все строки.")

file = st.file_uploader("Загрузить изображение", type=['jpg', 'jpeg', 'png'])

if file:
    img = Image.open(file)
    st.image(img, caption="Ваше фото", use_container_width=True)
    
    if st.button("🚀 Начать распознавание"):
        with st.spinner("Работают нейросети Google/Pytorch..."):
            # Конвертируем для EasyOCR
            img_np = np.array(img)
            reader = load_reader()
            
            # detail=0 вернет только текст списком
            results = reader.readtext(img_np, detail=0)
            
            # Собираем строки в один текст
            full_text = " ".join(results)
            
            if full_text:
                # Перевод
                translated = GoogleTranslator(source='auto', target='ru').translate(full_text)
                
                st.divider()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Распознанный текст")
                    st.write(full_text)
                
                with col2:
                    st.subheader("Перевод на русский")
                    st.success(translated)
                    
                # Усложнение 5 недели: Аналитика
                st.subheader("📊 Статистика")
                st.write(f"Найдено фрагментов текста: **{len(results)}**")
                st.write(f"Всего слов: **{len(full_text.split())}**")
            else:
                st.warning("Текст не найден. Попробуйте более четкое фото.")
