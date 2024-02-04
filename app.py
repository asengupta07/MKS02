import streamlit as st
import numpy as np
from PIL import Image
import io
from helper import predict

st.title('Student Engagement Detector')
st.write('Detecting student engagement in an online classroom setting using a deep learning CNN model.')
img = st.file_uploader('Upload an Image:', type=['jpg', 'png', 'jpeg'])
if img is not None:
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write('')
    st.write('Classifying...')
    image_np = np.frombuffer(img.read(), np.uint8)
    image_bytes = image_np.tobytes()
    image_file = io.BytesIO(image_bytes)
    img = Image.open(image_file)
    img = np.array(img)
    eng, conf = predict(img)
    st.write(f'Prediction: {eng}')
    st.write(f'Confidence: {conf*100:.2f}%')