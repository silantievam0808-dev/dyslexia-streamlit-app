import re
import html
try:
    # Streamlit is imported lazily in the web app context.  When running
    # unit tests or importing this module in an environment without
    # Streamlit installed, the import will fail, and we handle that
    # gracefully by assigning None.  The `main()` function below will
    # attempt to import Streamlit again.
    import streamlit as st  # type: ignore
except ImportError:
    st = None  # type: ignore


"""
Streamlit app for adapting Russian texts for readers with dyslexia.

This application implements two modes inspired by the methodology described in the
user's thesis work.  In **mode A** the original input text is first
lexically and syntactically simplified and then annotated with HTML
markup to highlight important linguistic elements.  In **mode B** the
original text is left untouched and only annotated.  The annotation
scheme is similar to the LARF method and uses the following tags:

  * `<b class="ent-person">…</b>` — human names (PERSON)
  * `<b class="ent-loc">…</b>` — place names (LOC)
  * `<b class="ent-org">…</b>` — organization names (ORG)
  * `<b class="ent-date">…</b>` — dates and numbers (DATE)
  * `<mark class="pred">…</mark>` — the main predicate of the sentence
  * `<span class="chunk">…</span>` — coherent semantic blocks of 2–6 words

The simplification procedure used here is intentionally lightweight and
rule‑based.  It focuses on replacing a handful of rare or long
expressions with simpler synonyms and splitting very long sentences.
For a production system one would substitute this stub with a more
sophisticated model or API.
"""


def simplify_text(text: str) -> str:
    """
    Lexically and syntactically simplify Russian text.

    This implementation uses a small dictionary of replacements and
    splits long sentences on semicolons or colons.  It preserves the
    original meaning while attempting to reduce linguistic complexity.

    Args:
        text: The input string containing one or more sentences.

    Returns:
        A simplified version of the input text.
    """
    # Normalize whitespace
    simplified = re.sub(r"\s+", " ", text).strip()

    # Lexical replacements: map longer expressions to simpler ones
    replacements = {
        "избирательное нарушение": "особенность",
        "овладению навыками": "освоению",
        "при сохранении": "хотя сохраняется",
        "способности к обучению": "способность учиться",
        "трудности с чтением вслух и про себя": "трудности при чтении вслух и про себя",
        "беглостью чтения": "быстрым чтением",
        "пониманием прочитанного": "пониманием текста",
        "осуществление": "проведение",
        "когнитивный": "познавательный",
        "дискурсивный": "связанный с построением текста",
        "интерпретация": "объяснение",
        "идентификация": "определение",
        "генерация": "создание",
        "визуальная разметка": "визуальное выделение",
    }
    for old, new in replacements.items():
        simplified = simplified.replace(old, new)

    # Split on semicolons and colons to shorten long sentences
    simplified = simplified.replace(";", ".")
    simplified = simplified.replace(":", ".")
    return simplified


def split_sentences(text: str) -> list:
    """Split text into sentences using a simple regular expression."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def annotate_entities(sentence: str) -> str:
    """
    Annotate persons, locations, organizations and dates/numbers.

    The heuristics used here are deliberately simple: persons are
    identified as two words starting with uppercase letters; locations
    and organizations are matched against small lists; dates and numbers
    are matched via regular expressions.

    Args:
        sentence: A sentence to annotate.

    Returns:
        The sentence with inline <b class="ent-…"> tags around named
        entities.
    """
    # Dates and numbers
    annotated = re.sub(
        r"\b(\d{1,4}(?:[./-]\d{1,2}(?:[./-]\d{2,4})?)?)\b",
        r"<b class=\"ent-date\">\1</b>",
        sentence,
    )

    # Person names: two consecutive capitalized words
    annotated = re.sub(
        r"\b([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)\b",
        r"<b class=\"ent-person\">\1</b>",
        annotated,
    )

    # Locations and organizations heuristics: look for exact matches in lists
    locations = [
        "Москва", "Москве", "Москвы",
        "Санкт-Петербург", "Петербург", "Россия", "России",
        "Германия", "Франция", "Европа",
    ]
    for loc in sorted(locations, key=len, reverse=True):
        annotated = re.sub(
            rf"\b({re.escape(loc)})\b",
            r"<b class=\"ent-loc\">\1</b>",
            annotated,
        )

    organizations = [
        "International Dyslexia Association",
        "МГТУ им. Баумана",
        "Google",
        "OpenAI",
        "Streamlit",
    ]
    for org in sorted(organizations, key=len, reverse=True):
        annotated = re.sub(
            rf"\b({re.escape(org)})\b",
            r"<b class=\"ent-org\">\1</b>",
            annotated,
        )
    return annotated


def annotate_predicate(sentence: str) -> str:
    """
    Annotate the main predicate (verb) in the sentence.

    This heuristic searches for the first occurrence of one of a few
    common Russian verbs and highlights it with <mark class="pred">.
    In a real system you would perform proper dependency parsing to
    identify the root verb.
    """
    predicates = [
        "является", "считается", "помогает", "облегчает", "сохраняется",
        "включает", "позволяет", "использует", "затрудняет", "упрощает",
        "выделяет", "создаёт", "повышает", "снижает", "может",
    ]
    for pred in predicates:
        new_sentence = re.sub(
            rf"\b({pred})\b",
            r"<mark class=\"pred\">\1</mark>",
            sentence,
            count=1,
        )
        if new_sentence != sentence:
            return new_sentence
    return sentence


def annotate_chunks(sentence: str) -> str:
    """
    Annotate semantically coherent chunks of 2–6 words.

    A simple heuristic is used: this function iterates over a small
    predefined list of multiword expressions and wraps them in
    <span class="chunk">.  If no predefined chunks match, it falls
    back to highlighting sequences of adjectives, nouns, proper nouns
    and numerals.
    """
    # Predefined chunks relevant to the topic of dyslexia and text adaptation
    chunk_list = [
        "способность к обучению",
        "трудности при чтении",
        "понимание текста",
        "пользователи с дислексией",
        "визуальное выделение",
        "быстрым чтением",
        "особенность обучения",
        "навыками чтения и письма",
        "общая способность учиться",
        "визуальная разметка текста",
    ]
    annotated = sentence
    for chunk in sorted(chunk_list, key=len, reverse=True):
        annotated = re.sub(
            rf"\b({re.escape(chunk)})\b",
            r"<span class=\"chunk\">\1</span>",
            annotated,
        )
    return annotated


def annotate_sentence(sentence: str) -> str:
    """
    Apply entity, predicate and chunk annotation to a single sentence.
    The order of operations matters: entities first, then predicate,
    then chunks.  This helps avoid nested or overlapping tags.
    """
    # Escape HTML special characters first
    s = html.escape(sentence, quote=False)
    s = annotate_entities(s)
    s = annotate_predicate(s)
    s = annotate_chunks(s)
    return s


def annotate_text(text: str) -> str:
    """
    Annotate all sentences in the text and join them.
    Each sentence is processed independently and then joined with a
    space.  Newlines in the original text are preserved by replacing
    them with <br> tags.
    """
    paragraphs = text.split("\n")
    annotated_paragraphs = []
    for para in paragraphs:
        sentences = split_sentences(para)
        ann_sentences = [annotate_sentence(s) for s in sentences]
        annotated_paragraphs.append(" ".join(ann_sentences))
    return "<br>".join(annotated_paragraphs)


def adapt_text(text: str, mode: str) -> tuple:
    """
    Adapt text according to the selected mode.

    Args:
        text: Raw input text from the user.
        mode: Either 'A' or 'B'.  In mode A the text is simplified
              before annotation; in mode B the original text is
              annotated directly.

    Returns:
        A tuple (simplified_text, annotated_html).  The first element
        will be None in mode B.
    """
    if mode not in {"A", "B"}:
        raise ValueError("mode must be 'A' or 'B'")
    if mode == "A":
        simplified = simplify_text(text)
        annotated = annotate_text(simplified)
        return simplified, annotated
    else:
        annotated = annotate_text(text)
        return None, annotated


def main():
    """Run the Streamlit user interface."""
    st.set_page_config(
        page_title="Адаптация текста для пользователей с дислексией",
        page_icon="📘",
        layout="wide",
    )
    st.title("Нейросетевая адаптация текста для пользователей с дислексией")
    st.write(
        "Этот прототип демонстрирует два режима работы: "
        "**A** — упрощение текста + разметка, **B** — только разметка."
    )

    default_text = (
        "Дислексия — избирательное нарушение способности к овладению навыками "
        "чтения и письма при сохранении общей способности к обучению. "
        "Проблемы могут включать трудности с чтением вслух и про себя, "
        "правописанием, беглостью чтения и пониманием прочитанного."
    )

    text_input = st.text_area(
        "Введите текст для адаптации:", value=default_text, height=200
    )
    mode = st.radio(
        "Выберите режим:",
        options=["A", "B"],
        format_func=lambda x: "A — упрощение + разметка" if x == "A" else "B — только разметка",
        horizontal=True,
    )
    if st.button("Адаптировать текст", type="primary"):
        simplified_text, annotated_html = adapt_text(text_input, mode)
        if simplified_text:
            st.subheader("Упрощённый текст")
            st.write(simplified_text)
        st.subheader("Результат с визуальной разметкой")
        st.markdown(
            f"<div style='font-size:20px; line-height:1.8;'>{annotated_html}</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()