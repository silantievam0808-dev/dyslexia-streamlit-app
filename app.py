import streamlit as st

st.title("Прототип адаптации текста для дислексии")

text = st.text_area("Введите текст")

if st.button("Показать"):
    st.write(text)
