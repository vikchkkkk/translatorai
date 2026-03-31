import io
import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from deep_translator import GoogleTranslator

# --- КОНФИГУРАЦИЯ СТРАНИЦЫ ---
st.set_page_config(page_title="Polyglot OCR УрФУ", page_icon="📝")

# --- ОБЯЗАТЕЛЬНАЯ ИНФОРМАЦИЯ (Неделя 4) ---
with st.sidebar:
    st.title("О проекте")
    st.markdown("""
    **Программа:** [Цифровые кафедры УрФУ: Python в гуманитаристике](https://dpo.urfu.ru/programs/92)
    
    **Команда проекта:**
    * **Руководитель:** [Твое Имя]
    * **Администратор:** [Имя участника]
    * **Программист:** [Имя участника]
    * **Аналитик:** [Имя участника]
    
    **Технологии:**
    * Модель: `TrOCR-base-printed`
    * Библиотеки: `transformers`, `torch`, `streamlit`
    """)
    st.divider()
    st.info("Совет: Для лучшего результата загружайте изображения с одной строкой текста.")

# --- ЗАГРУЗКА МОДЕЛИ ---
@st.cache_resource
def load_ocr_model():
    # Ручная загрузка процессора и модели для стабильности на Python 3.14
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    return processor, model

# --- ГЛАВНЫЙ ИНТЕРФЕЙС ---
st.title("📝 Polyglot OCR: Распознавание и Перевод")
st.write("Приложение для автоматического извлечения текста (EN/FR) и перевода на русский.")

uploaded_file = st.file_uploader("Выберите изображение (желательно одну строку текста)", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Загрузка и конвертация в RGB (важно для нейросети!)
    image_data = uploaded_file.getvalue()
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    st.image(img, caption="Загруженное изображение", use_container_width=True)
    
    if st.button('🚀 Распознать и перевести'):
        with st.spinner('Нейросеть анализирует пиксели...'):
            try:
                # 1. OCR процесс
                processor, model = load_ocr_model()
                pixel_values = processor(images=img, return_tensors="pt").pixel_values
                generated_ids = model.generate(pixel_values)
                recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                if not recognized_text.strip():
                    recognized_text = "[Текст не обнаружен. Попробуйте обрезать изображение до одной строки]"

                # 2. Перевод (Усложнение недели 5)
                translated_text = GoogleTranslator(source='auto', target='ru').translate(recognized_text)

                # --- ВЫВОД РЕЗУЛЬТАТОВ ---
                st.divider()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Оригинал (EN/FR)")
                    st.code(recognized_text)
                
                with col2:
                    st.subheader("Перевод (RU)")
                    st.success(translated_text)

                # Статистика (Усложнение недели 5)
                st.subheader("📊 Аналитика")
                st.write(f"Количество символов: **{len(recognized_text)}**")
                st.write(f"Количество слов: **{len(recognized_text.split())}**")

            except Exception as e:
                st.error(f"Произошла ошибка: {e}")

# --- МАТЕМАТИЧЕСКАЯ СПРАВКА ---
with st.expander("Посмотреть формулу работы"):
    st.latex(r"Y = \text{argmax}_{y} P(y | X; \theta)")
    st.write("Где X — пиксели изображения, а Y — генерируемый текст.")
