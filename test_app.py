import streamlit as st

st.title('Anemia Detection System - Test')
st.write('This is a test deployment')

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
if uploaded_file:
    st.image(uploaded_file)
