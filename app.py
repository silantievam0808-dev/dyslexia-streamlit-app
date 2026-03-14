import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
from natasha import (
    Segmenter, MorphVocab, NewsEmbedding, NewsNERTagger, Doc
)
import torch

# Настройка страницы
st.set_page_config(page_title="Dyslexia Adapt", layout="wide")

# Кастомный CSS для разметки (согласно вашей работе)
st.markdown("""
<style>
    .per { background-color: #d1e9ff; border-radius: 3px; padding: 2px; font-weight: bold; }
    .loc { background-color: #ffdee9; border-radius: 3px; padding: 2px; font-weight: bold; }
    .org { background-color: #e2f0d9; border-radius: 3px; padding: 2px; font-weight: bold; }
    .main-text { line-height: 1.8; font-size: 1.2rem; letter-spacing: 0.05em; }
</style>
""", unsafe_allow_html=True)

# Загрузка легкой модели упрощения
@st.cache_resource
def load_models():
    model_name = "cointegrated/rut5-base-paraphraser"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Инициализация Natasha
    segmenter = Segmenter()
    ner_tagger = NewsNERTagger(NewsEmbedding())
    morph_vocab = MorphVocab()
    
    return tokenizer, model, segmenter, ner_tagger, morph_vocab

tokenizer, model, segmenter, ner_tagger, morph_vocab = load_models()

def simplify_text(text):
    """Генеративное упрощение (Режим A)"""
    inputs = tokenizer(text, return_tensors='pt', padding=True)
    with torch.no_grad():
        out = model.generate(
            **inputs, 
            max_length=150, 
            do_sample=False, 
            repetition_penalty=2.5
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

def apply_markup(text):
    """Визуальная разметка сущностей (Методология LARF)"""
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    
    # Сортируем сущности с конца, чтобы не нарушить индексы при замене
    spans = sorted(doc.spans, key=lambda x: x.start, reverse=True)
    
    html_text = text
    for span in spans:
        label_class = span.type.lower() # per, loc, org
        start, stop = span.start, span.stop
        tagged_part = f'<span class="{label_class}">{text[start:stop]}</span>'
        html_text = html_text[:start] + tagged_part + html_text[stop:]
    
    return html_text

# Интерфейс
st.title("Система адаптации текста")
st.write("Инструмент поддержки чтения на основе упрощения и визуальной разметки.")

mode = st.radio(
    "Выберите режим методологии:",
    ["Режим A (Упрощение + Разметка)", "Режим B (Только разметка LARF)"]
)

input_text = st.text_area("Введите исходный текст:", height=200)

if st.button("Адаптировать"):
    if input_text:
        with st.spinner("Обработка..."):
            if "Режим A" in mode:
                # Сначала упрощаем
                processed = simplify_text(input_text)
                # Затем размечаем упрощенный текст
                final_html = apply_markup(processed)
            else:
                # Режим B: только разметка оригинала
                final_html = apply_markup(input_text)
            
            st.subheader("Адаптированный текст:")
            st.markdown(f'<div class="main-text">{final_html}</div>', unsafe_allow_html=True)
    else:
        st.warning("Введите текст для обработки.")
