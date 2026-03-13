import html
import re
from typing import List, Tuple

import streamlit as st
from razdel import sentenize, tokenize
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsNERTagger, Doc
import pymorphy2


st.set_page_config(
    page_title="Адаптация текста для пользователей с дислексией",
    page_icon="📘",
    layout="wide"
)

CUSTOM_CSS = """
<style>
.result-box {
    font-size: 22px;
    line-height: 1.9;
    background: #ffffff;
    padding: 24px;
    border-radius: 16px;
    border: 1px solid #d9d9d9;
    margin-top: 12px;
}
.ent-person {
    font-weight: 700;
    color: #8e24aa;
}
.ent-loc {
    font-weight: 700;
    color: #1565c0;
}
.ent-org {
    font-weight: 700;
    color: #2e7d32;
}
.ent-date {
    font-weight: 700;
    color: #ef6c00;
}
.pred {
    background-color: #fff59d;
    padding: 2px 4px;
    border-radius: 4px;
}
.chunk {
    background-color: #f5f5f5;
    padding: 2px 4px;
    border-radius: 4px;
}
.legend-box {
    background: #fafafa;
    border: 1px solid #e5e5e5;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
}
.small-note {
    color: #666;
    font-size: 15px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
ner_tagger = NewsNERTagger(emb)
morph = pymorphy2.MorphAnalyzer()


LEXICAL_REPLACEMENTS = {
    "избирательное нарушение": "особенность",
    "овладение навыками": "освоение",
    "овладению навыками": "освоению",
    "при сохранении": "хотя сохраняется",
    "способности к обучению": "способность учиться",
    "правописание": "написание слов без ошибок",
    "беглость чтения": "быстрое чтение",
    "понимание прочитанного": "понимание текста",
    "когнитивная нагрузка": "умственная нагрузка",
    "визуальная разметка": "визуальное выделение",
    "генерация": "создание",
    "идентификация": "определение",
    "интерпретация": "объяснение",
    "дискурсивный": "связанный с построением текста",
    "осуществление": "проведение",
    "низкочастотный": "редкий",
    "лексическая единица": "слово",
    "номинализация": "отглагольное существительное",
    "причинно-следственные связи": "связи между причиной и следствием",
}


DISCOURSE_MARKERS = {
    "однако", "поэтому", "потому что", "следовательно", "таким образом",
    "например", "кроме того", "при этом", "в результате", "сначала", "затем"
}


def normalize_spaces(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_long_sentence(sentence: str, max_words: int = 16) -> str:
    words = sentence.split()
    if len(words) <= max_words:
        return sentence

    separators = [", потому что ", ", так как ", ", однако ", ", но ", ", а также ", "; ", ": "]
    for sep in separators:
        if sep in sentence:
            parts = sentence.split(sep, 1)
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                joiner = ". " if not right[:1].islower() else ". "
                return left + joiner + right[:1].upper() + right[1:] if right else left
    return sentence


def simplify_sentence(sentence: str) -> str:
    s = sentence.strip()

    for old, new in LEXICAL_REPLACEMENTS.items():
        s = re.sub(rf"\b{re.escape(old)}\b", new, s, flags=re.IGNORECASE)

    s = re.sub(
        r"Проблемы могут включать (.+)",
        r"У человека могут быть такие трудности: \1",
        s,
        flags=re.IGNORECASE
    )

    s = re.sub(
        r"которые затрагивают точность, скорость или оба этих аспекта",
        "которые влияют на точность и скорость чтения",
        s,
        flags=re.IGNORECASE
    )

    s = re.sub(
        r"варьируются в зависимости от особенностей орфографии",
        "и проявляются по-разному в разных языках",
        s,
        flags=re.IGNORECASE
    )

    s = split_long_sentence(s)

    s = re.sub(r"\s+,", ",", s)
    s = re.sub(r"\s+\.", ".", s)
    s = re.sub(r"\s+:", ":", s)
    s = re.sub(r"\s+;", ";", s)
    s = re.sub(r"\s{2,}", " ", s)

    return s.strip()


def simplify_text(text: str) -> str:
    sentences = [s.text.strip() for s in sentenize(normalize_spaces(text))]
    simplified = [simplify_sentence(s) for s in sentences if s.strip()]
    result = " ".join(simplified)
    result = re.sub(r"\s{2,}", " ", result)
    return result.strip()


def get_entities(sentence: str) -> List[Tuple[int, int, str]]:
    doc = Doc(sentence)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)

    spans = []
    for span in doc.spans:
        label = None
        if span.type == "PER":
            label = "ent-person"
        elif span.type == "LOC":
            label = "ent-loc"
        elif span.type == "ORG":
            label = "ent-org"
        elif span.type == "DATE":
            label = "ent-date"

        if label:
            spans.append((span.start, span.stop, label))

    for match in re.finditer(r"\b\d{1,4}(?:[./-]\d{1,2}(?:[./-]\d{2,4})?)?\b", sentence):
        spans.append((match.start(), match.end(), "ent-date"))

    spans = sorted(set(spans), key=lambda x: (x[0], x[1]))
    return spans


def is_verb(token_text: str) -> bool:
    parsed = morph.parse(token_text)
    if not parsed:
        return False
    tag = parsed[0].tag
    return "VERB" in tag or "INFN" in tag


def is_good_chunk_token(token_text: str) -> bool:
    parsed = morph.parse(token_text)
    if not parsed:
        return False
    pos = parsed[0].tag.POS
    return pos in {"NOUN", "ADJF", "ADJS", "PRTF", "PRTS", "NUMR"}


def get_predicate_span(sentence: str) -> Tuple[int, int] | None:
    tokens = list(tokenize(sentence))

    for token in tokens:
        t = token.text
        if is_verb(t):
            parsed = morph.parse(t)[0]
            if parsed.tag.POS == "INFN":
                continue
            return token.start, token.stop

    for token in tokens:
        if token.text.lower() in {"может", "могут", "нужно", "следует", "является", "бывает"}:
            return token.start, token.stop

    return None


def get_chunk_spans(sentence: str) -> List[Tuple[int, int]]:
    tokens = list(tokenize(sentence))
    chunks = []
    current = []

    for token in tokens:
        text = token.text

        if text.lower() in DISCOURSE_MARKERS:
            if current:
                chunks.append((current[0].start, current[-1].stop))
                current = []
            continue

        if is_good_chunk_token(text):
            current.append(token)
        else:
            if current:
                if 1 <= len(current) <= 6:
                    chunks.append((current[0].start, current[-1].stop))
                current = []

    if current and 1 <= len(current) <= 6:
        chunks.append((current[0].start, current[-1].stop))

    filtered = []
    for start, end in chunks:
        chunk_text = sentence[start:end].strip()
        if len(chunk_text.split()) >= 2:
            filtered.append((start, end))

    return filtered


def overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def build_html(sentence: str) -> str:
    entities = get_entities(sentence)
    pred = get_predicate_span(sentence)
    chunks = get_chunk_spans(sentence)

    labels = [None] * len(sentence)

    for start, end, label in entities:
        for i in range(start, min(end, len(sentence))):
            labels[i] = label

    if pred:
        p_start, p_end = pred
        for i in range(p_start, min(p_end, len(sentence))):
            if labels[i] is None:
                labels[i] = "pred"

    for c_start, c_end in chunks:
        if pred and overlaps(c_start, c_end, pred[0], pred[1]):
            continue
        if any(label and label.startswith("ent-") for label in labels[c_start:c_end]):
            continue
        for i in range(c_start, min(c_end, len(sentence))):
            if labels[i] is None:
                labels[i] = "chunk"

    parts = []
    current = None

    open_tags = {
        "ent-person": '<b class="ent-person">',
        "ent-loc": '<b class="ent-loc">',
        "ent-org": '<b class="ent-org">',
        "ent-date": '<b class="ent-date">',
        "pred": '<mark class="pred">',
        "chunk": '<span class="chunk">',
    }

    close_tags = {
        "ent-person": "</b>",
        "ent-loc": "</b>",
        "ent-org": "</b>",
        "ent-date": "</b>",
        "pred": "</mark>",
        "chunk": "</span>",
    }

    for i, ch in enumerate(sentence):
        label = labels[i]
        if label != current:
            if current is not None:
                parts.append(close_tags[current])
            if label is not None:
                parts.append(open_tags[label])
            current = label
        parts.append(html.escape(ch, quote=False))

    if current is not None:
        parts.append(close_tags[current])

    return "".join(parts)


def annotate_text(text: str) -> str:
    sentences = [s.text.strip() for s in sentenize(normalize_spaces(text))]
    rendered = [build_html(s) for s in sentences if s.strip()]
    return " ".join(rendered)


def adapt_text(text: str, mode: str) -> Tuple[str | None, str]:
    source = normalize_spaces(text)

    if mode == "A":
        simplified = simplify_text(source)
        marked = annotate_text(simplified)
        return simplified, marked

    marked = annotate_text(source)
    return None, marked


st.title("Адаптация текста для пользователей с дислексией")
st.markdown(
    """
<div class="legend-box">
<b>Режим A</b>: сначала генерируется упрощённый текст, затем к нему применяется визуальная разметка.<br>
<b>Режим B</b>: на исходный текст накладывается только разметка, без генеративного переписывания.
</div>
""",
    unsafe_allow_html=True,
)

default_text = """Дислексия — избирательное нарушение способности к овладению навыками чтения и письма при сохранении общей способности к обучению. Проблемы могут включать трудности с чтением вслух и про себя, правописанием, беглостью чтения и пониманием прочитанного."""

text = st.text_area("Вставьте текст", value=default_text, height=220)

mode = st.radio(
    "Выберите режим",
    options=["A", "B"],
    format_func=lambda x: "Режим A — упрощение + разметка" if x == "A" else "Режим B — только разметка",
)

if st.button("Адаптировать текст", type="primary"):
    simplified_text, marked_html = adapt_text(text, mode)

    if simplified_text is not None:
        st.subheader("Упрощённый текст")
        st.write(simplified_text)

    st.subheader("Результат с визуальной разметкой")
    st.markdown(f'<div class="result-box">{marked_html}</div>', unsafe_allow_html=True)

st.markdown(
    """
<p class="small-note">
Текущий MVP использует облегчённое rule-based упрощение и лингвистическую разметку.
Это стабильнее для Streamlit Cloud, чем тяжёлые трансформерные модели.
</p>
""",
    unsafe_allow_html=True,
)
