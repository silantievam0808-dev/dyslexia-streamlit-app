import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
from natasha import (
    Segmenter, MorphVocab, NewsEmbedding, NewsNERTagger,
    NamesExtractor, DatesExtractor, Doc
)
import re

# --- ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ (Кэшируем, чтобы не загружать каждый раз) ---
@st.cache_resource
def load_simplification_model():
    # Используем базовую модель ruT5 как placeholder для вашего будущего fine-tuned FRED-T5
    model_name = "cointegrated/rut5-base-paraphraser"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def load_natasha_ner():
    segmenter = Segmenter()
    emb = NewsEmbedding()
    ner_tagger = NewsNERTagger(emb)
    morph_vocab = MorphVocab()
    return segmenter, ner_tagger, morph_vocab

# --- ЛОГИКА УПРОЩЕНИЯ (РЕЖИМ A) ---
def simplify_text(text, tokenizer, model):
    # Разделение текста на предложения (в идеале использовать razdel, как указано в ВКР)
    sentences = re.split(r'(?<=[.!?]) +', text)
    simplified_sentences = []
    
    for sentence in sentences:
        if len(sentence) < 3:
            continue
        # Генерация упрощенного текста (T5)
        inputs = tokenizer(sentence, return_tensors="pt", padding=True)
        outputs = model.generate(
            **inputs, 
            max_length=100, 
            num_beams=3, 
            repetition_penalty=2.5,
            early_stopping=True
        )
        simplified = tokenizer.decode(outputs[0], skip_special_tokens=True)
        simplified_sentences.append(simplified)
        
    return " ".join(simplified_sentences)

# --- ЛОГИКА РАЗМЕТКИ LARF (РЕЖИМ B И ЧАСТЬ РЕЖИМА A) ---
def apply_larf_markup(text, segmenter, ner_tagger, morph_vocab):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    
    # Нормализация сущностей
    for span in doc.spans:
        span.normalize(morph_vocab)
        
    # Сортируем спаны с конца, чтобы при замене индексы не сдвигались
    spans = sorted(doc.spans, key=lambda x: x.start, reverse=True)
    
    marked_text = text
    for span in spans:
        # Применяем HTML-теги из вашей таблицы 6
        if span.type == 'PER':
            tag_open, tag_close = '<b class="ent-person" style="background-color: #e6f7ff;">', '</b>'
        elif span.type == 'LOC':
            tag_open, tag_close = '<b class="ent-loc" style="background-color: #fff0f6;">', '</b>'
        elif span.type == 'ORG':
            tag_open, tag_close = '<b class="ent-org" style="background-color: #f6ffed;">', '</b>'
        else:
            tag_open, tag_close = '<b>', '</b>'
            
        marked_text = marked_text[:span.start] + tag_open + marked_text[span.start:span.stop] + tag_close + marked_text[span.stop:]
        
    # Пример добавления <mark> для глаголов или смысловых блоков можно реализовать через морфологический анализ
    return marked_text

# --- ИНТЕРФЕЙС STREAMLIT ---
st.set_page_config(page_title="Адаптация текстов для дислексиков", layout="wide")

st.title("Нейросетевая система адаптации текстов (LARF)")
st.markdown("Инструмент для лингвистической адаптации и визуальной поддержки чтения[cite: 226].")

# Загрузка моделей
with st.spinner('Загрузка нейросетевых моделей...'):
    tokenizer, model = load_simplification_model()
    segmenter, ner_tagger, morph_vocab = load_natasha_ner()

# Выбор режима
mode = st.radio(
    "Выберите режим работы системы:",
    ("Режим A: Генеративное упрощение + Визуальная разметка", 
     "Режим B: Только визуальная разметка (LARF)")
)

source_text = st.text_area("Введите текст для адаптации:", height=200)

if st.button("Адаптировать текст"):
    if source_text.strip():
        with st.spinner("Обработка текста..."):
            if "Режим A" in mode:
                # 1. Упрощение
                simplified_text = simplify_text(source_text, tokenizer, model)
                # 2. Разметка
                final_text = apply_larf_markup(simplified_text, segmenter, ner_tagger, morph_vocab)
                
                st.subheader("Результат (Режим A):")
                st.markdown(final_text, unsafe_allow_html=True)
                
            else:
                # Только разметка исходного текста
                final_text = apply_larf_markup(source_text, segmenter, ner_tagger, morph_vocab)
                
                st.subheader("Результат (Режим B - LARF):")
                st.markdown(final_text, unsafe_allow_html=True)
    else:
        st.warning("Пожалуйста, введите текст.")
