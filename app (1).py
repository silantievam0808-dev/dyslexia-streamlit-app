"""
Streamlit application for adapting Russian text for readers with dyslexia.

This version provides a more robust approach to both simplification and
visual annotation.  It leverages a neural text simplification model
(`ruT5-base-multitask`) to generate a simplified version of the input
text and uses the `Natasha` library to extract named entities,
identify the root verb (predicative), and group noun phrases into
semantic chunks.  The resulting HTML is styled with CSS classes to
highlight persons, locations, organizations, dates, predicates and
chunks, following the LARF methodology【216500612612523†L745-L767】.

The app operates in two modes:

  * **Mode A (simplification + annotation):** The text is first
    simplified by the model and then annotated.
  * **Mode B (annotation only):** The original text is left
    untouched and only annotated.  This mode minimizes the risk of
    semantic drift【216500612612523†L745-L767】.

To run this app locally or deploy it on Streamlit Community Cloud,
ensure that the dependencies listed in `requirements.txt` are
installed.  In a resource‑constrained environment (e.g. without GPU
access) the model will run on CPU; expect some delay for long texts.
"""

import re
import html
import torch
from typing import Tuple

# Lazy import of optional libraries.  If these packages are not
# installed, an ImportError will be thrown at runtime when the
# respective functions are called.  This allows unit testing of
# non‑dependent code without installing heavy dependencies.
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    AutoTokenizer = None  # type: ignore
    AutoModelForSeq2SeqLM = None  # type: ignore

try:
    from natasha import (
        Segmenter,
        MorphVocab,
        NewsEmbedding,
        NewsMorphTagger,
        NewsNERTagger,
        Doc,
    )
except ImportError:
    Segmenter = MorphVocab = NewsEmbedding = NewsMorphTagger = None  # type: ignore
    NewsNERTagger = Doc = None  # type: ignore

# Streamlit is imported inside main() to allow importing this module
# without having streamlit installed (e.g. during testing).


class TextAdapter:
    """
    Encapsulates the models and methods for simplifying and annotating text.
    """

    def __init__(self):
        # Initialize ruT5 model for text simplification
        if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
            raise ImportError(
                "transformers is required for simplification. Please install it via pip."
            )
        model_name = "cointegrated/rut5-base-multitask"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Initialize Natasha for NER and morphological analysis
        if Segmenter is None:
            raise ImportError(
                "Natasha library is required for annotation. Please install it via pip."
            )
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(emb)
        self.ner_tagger = NewsNERTagger(emb)

    def simplify_text(self, text: str, max_length: int = 512, num_beams: int = 5) -> str:
        """
        Generate a simplified version of Russian text using ruT5.

        The model is invoked with the "simplify | " prefix as
        documented for the multitask version【216972154572502†L39-L48】.

        Args:
            text: The input text.
            max_length: Maximum length of the generated sequence.
            num_beams: Beam width for beam search.

        Returns:
            A simplified version of the input text.
        """
        clean = re.sub(r"\s+", " ", text).strip()
        # Add the multitask prefix for simplification
        prefix = "simplify | "
        input_ids = self.tokenizer(
            prefix + clean,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **input_ids,
                max_length=max_length,
                num_beams=num_beams,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )
        simplified = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return simplified.strip()

    def annotate_text(self, text: str) -> str:
        """
        Annotate Russian text with HTML tags following the LARF scheme.

        Steps:
          1. Segment text into sentences and run morphological and NER
             analysis via Natasha.
          2. For each sentence, assign labels to characters:
             - PERSON, LOC, ORG, DATE from NER
             - predicate (root verb)
             - noun phrase (chunk) via POS patterns
             - mark summarising or unusual sentences
          3. Insert HTML tags at character boundaries to wrap
             segments.

        Args:
            text: A string containing one or more sentences.

        Returns:
            Annotated HTML string.
        """
        # Split by paragraphs to preserve newlines
        paragraphs = text.split("\n")
        annotated_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                annotated_paragraphs.append("")
                continue
            doc = Doc(para)
            doc.segment(self.segmenter)
            doc.tag_morph(self.morph_tagger)
            doc.tag_ner(self.ner_tagger)
            sentences = list(doc.sents)
            total = len(sentences)
            annotated_sents = []
            for idx, sent in enumerate(sentences):
                sent_text = sent.text
                length = len(sent_text)
                labels: list[str | None] = [None] * length
                # 1. NER entities (PERSON, LOC, ORG, DATE)
                for span in doc.spans:
                    if span.start >= sent.start and span.stop <= sent.stop:
                        if span.type in {"PER", "LOC", "ORG", "DATE"}:
                            start = span.start - sent.start
                            end = span.stop - sent.start
                            cls = None
                            if span.type == "PER":
                                cls = "ent-person"
                            elif span.type == "LOC":
                                cls = "ent-loc"
                            elif span.type == "ORG":
                                cls = "ent-org"
                            elif span.type == "DATE":
                                cls = "ent-date"
                            if cls:
                                for i in range(start, end):
                                    labels[i] = cls
                # 2. Predicate (root verb)
                root = sent.root
                if root.pos in {"VERB", "AUX"}:
                    start = root.idx - sent.start_char
                    end = start + len(root.text)
                    for i in range(start, end):
                        if labels[i] is None:
                            labels[i] = "pred"
                # 3. Noun phrases (chunk)
                chunk_tokens = []
                chunk_spans = []
                def flush_chunk():
                    nonlocal chunk_tokens, chunk_spans
                    if chunk_tokens:
                        cs = chunk_tokens[0].idx - sent.start_char
                        ce = (chunk_tokens[-1].idx - sent.start_char) + len(chunk_tokens[-1].text)
                        chunk_spans.append((cs, ce))
                        chunk_tokens = []
                for tok in sent:
                    if tok.pos in {"ADJ", "NOUN", "PROPN", "NUM"}:
                        chunk_tokens.append(tok)
                    else:
                        flush_chunk()
                flush_chunk()
                for start, end in chunk_spans:
                    for i in range(start, end):
                        if labels[i] is None:
                            labels[i] = "chunk"
                # 4. Summarising sentences (mark) and unusual (underline)
                summary_keywords = {
                    "итог", "вывод", "следовательно", "таким образом",
                    "в заключение", "результат", "таким образом",
                }
                mark_sentence = (
                    idx == 0 or idx == total - 1 or any(kw in sent_text.lower() for kw in summary_keywords)
                )
                underline_sentence = bool(re.search(r"[!?«»—]", sent_text))
                # 5. Build HTML with nested tags
                html_parts: list[str] = []
                current_label: str | None = None
                for pos, ch in enumerate(sent_text):
                    lbl = labels[pos]
                    # close tags if necessary
                    if lbl != current_label:
                        if current_label is not None:
                            if current_label.startswith("ent"):
                                html_parts.append("</b>")
                            elif current_label == "pred":
                                html_parts.append("</mark>")
                            elif current_label == "chunk":
                                html_parts.append("</span>")
                        # open new tag
                        if lbl is not None:
                            if lbl.startswith("ent"):
                                html_parts.append(f'<b class="{lbl}">')
                            elif lbl == "pred":
                                html_parts.append('<mark class="pred">')
                            elif lbl == "chunk":
                                html_parts.append('<span class="chunk">')
                        current_label = lbl
                    html_parts.append(html.escape(ch, quote=False))
                # close last tag
                if current_label is not None:
                    if current_label.startswith("ent"):
                        html_parts.append("</b>")
                    elif current_label == "pred":
                        html_parts.append("</mark>")
                    elif current_label == "chunk":
                        html_parts.append("</span>")
                annotated = "".join(html_parts)
                # wrap summary / underline
                if mark_sentence:
                    annotated = f"<mark>{annotated}</mark>"
                if underline_sentence:
                    annotated = f"<u>{annotated}</u>"
                annotated_sents.append(annotated)
            annotated_paragraphs.append(" ".join(annotated_sents))
        return "<br>".join(annotated_paragraphs)

    def adapt(self, text: str, mode: str) -> Tuple[str | None, str]:
        """
        Adapt text according to the specified mode.

        Args:
            text: Input text.
            mode: "A" or "B".  "A" triggers simplification before
                  annotation; "B" annotates the original text.

        Returns:
            A tuple (simplified_text, annotated_html).  The first
            element is None when mode B is selected.
        """
        if mode not in {"A", "B"}:
            raise ValueError("mode must be 'A' or 'B'")
        if mode == "A":
            simplified = self.simplify_text(text)
            annotated = self.annotate_text(simplified)
            return simplified, annotated
        else:
            annotated = self.annotate_text(text)
            return None, annotated


def main():
    # Import streamlit here to avoid a hard dependency when the module is imported
    import streamlit as st  # type: ignore

    st.set_page_config(
        page_title="Адаптация текста для пользователей с дислексией",
        page_icon="📘",
        layout="wide",
    )
    st.title("Нейросетевая адаптация текста для пользователей с дислексией")
    st.markdown(
        "Этот прототип демонстрирует два режима работы:\n"
        "**A** — сначала упрощение текста и затем разметка,\n"
        "**B** — только разметка (без изменения исходного текста)."
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
        # Instantiate the adapter on demand to avoid long start time
        with st.spinner("Загрузка моделей и анализ текста..."):
            adapter = TextAdapter()
            simplified, annotated_html = adapter.adapt(text_input, mode)
        if simplified:
            st.subheader("Упрощённый текст")
            st.write(simplified)
        st.subheader("Результат с визуальной разметкой")
        st.markdown(
            f"<div style='font-size:20px; line-height:1.8'>{annotated_html}</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()