import streamlit as st
import numpy as np
from PIL import Image
import io
from helper import predict

st.title('Student Engagement Detector')
st.write('Detecting student engagement in an online classroom setting using a deep learning CNN model.')
st.markdown("""
## Due to resource limitations in Streamlit, we have deployed this demo on Huggingface instead.
### [Link to Huggingface Spaces](https://huggingface.co/spaces/asengupta07/EngagementTracker)
""")

# img = st.file_uploader('Upload an Image:', type=['jpg', 'png', 'jpeg'])
# st.write('Here are some sample images to try out the model:')
# col1, col2, col3 = st.columns(3)
# with col1:
#     eg1 = st.image('images/example1.jpg', caption='Example 1', use_column_width=True)
#     if st.button('Try Example 1'):
#         img = open('images/example1.jpg', 'rb')
#     eg4 = st.image('images/example4thumb.jpg', caption='Example 4', use_column_width=True)
#     if st.button('Try Example 4'):
#         img = open('images/example4.jpg', 'rb')
# with col2:
#     eg2 = st.image('images/example2thumb.jpg', caption='Example 2', use_column_width=True)
#     if st.button('Try Example 2'):
#         img = open('images/example2.jpg', 'rb')
#     eg5 = st.image('images/example5thumb.jpg', caption='Example 5', use_column_width=True)
#     if st.button('Try Example 5'):
#         img = open('images/example5.jpg', 'rb')
# with col3:
#     eg3 = st.image('images/example3.jpg', caption='Example 3', use_column_width=True)
#     if st.button('Try Example 3'):
#         img = open('images/example3.jpg', 'rb')
#     eg6 = st.image('images/example6thumb.jpg', caption='Example 6', use_column_width=True)
#     if st.button('Try Example 6'):
#         img = open('images/example6.jpg', 'rb')

# if img is not None:
#         st.write('')
#         image_np = np.frombuffer(img.read(), np.uint8)
#         image_bytes = image_np.tobytes()
#         image_file = io.BytesIO(image_bytes)
#         img = Image.open(image_file)
#         img = np.array(img)
#         st.subheader('Result:')
#         st.image(img, caption='Uploaded Image', use_column_width=True)
#         eng, conf = predict(img)
#         st.subheader('Classification Report')
#         st.write(f'Prediction: {eng}')
#         st.write(f'Confidence: {conf*100:.2f}%')
