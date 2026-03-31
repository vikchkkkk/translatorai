import io
import streamlit as st
from transformers import pipeline
from PIL import Image
from deep_translator import GoogleTranslator

# Настройка страницы
st.set_page_config(page_title="Polyglot OCR", page_icon="🌍")

@st.cache_resource
def load_ocr_model():
    # Заменяем "image-to-text" на "image-text-to-text"
    return pipeline("image-text-to-text", model="microsoft/trocr-base-printed")

def get_text_metrics(text):
    """Усложнение: Аналитика текста"""
    chars = len(text)
    words = len(text.split())
    # Считаем среднюю длину слова для оценки сложности
    avg_word_len = chars / words if words > 0 else 0
    return {
        "chars": chars,
        "words": words,
        "complexity": "Высокая" if avg_word_len > 6 else "Средняя/Низкая"
    }

# --- ИНТЕРФЕЙС ---
st.title('🌍 Polyglot OCR: EN/FR to RU')
st.markdown("Приложение распознает английский или французский текст и переводит его на русский.")

uploaded_file = st.file_uploader(label='Загрузите фото с текстом (EN/FR)', type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image_data = uploaded_file.getvalue()
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    st.image(img, caption='Ваше изображение', width=400)
    
    if st.button('🔍 Распознать и перевести'):
        with st.spinner('Машинное зрение работает...'):
            # 1. OCR (Распознавание)
            model = load_ocr_model()
            ocr_result = model(img)
            recognized_text = ocr_result[0]["generated_text"]
            
            # 2. Перевод
            translated_text = GoogleTranslator(source='auto', target='ru').translate(recognized_text)
            
            # 3. Метрики
            metrics = get_text_metrics(recognized_text)
            
            # Вывод
            st.divider()
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📝 Распознанный текст")
                st.info(recognized_text)
                
            with col2:
                st.subheader("🇷🇺 Перевод")
                st.success(translated_text)
            
            # Дополнительный модуль (Усложнение)
            st.subheader("📊 Анализ текста")
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Символов", metrics["chars"])
            m_col2.metric("Слов", metrics["words"])
            m_col3.metric("Сложность", metrics["complexity"])

    # Техническая справка (LaTeX)
    with st.expander("Технические подробности"):
        st.write("Используемая архитектура: Vision Encoder-Decoder (TrOCR)")
        st.latex(r"Text_{out} = \text{Decoder}(\text{Encoder}(Image))")
