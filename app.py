import streamlit as st
st.set_page_config(page_title="Anemia Detection", layout="wide")

import os
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import mediapipe as mp

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'conjunctiva_region' not in st.session_state:
    st.session_state.conjunctiva_region = None

@st.cache_resource
def load_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

face_mesh = load_face_mesh()

def detect_conjunctiva(image):
    try:
        image = image.convert('RGB')
        image_array = np.array(image)
        height, width = image_array.shape[:2]
        
        results = face_mesh.process(image_array)
        
        if not results.multi_face_landmarks:
            st.warning("Eye landmarks not detected. Please ensure the eye is clearly visible.")
            return None, None
        
        # Lower eyelid landmarks
        lower_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133]
        
        points = []
        for idx in lower_eye_indices:
            landmark = results.multi_face_landmarks[0].landmark[idx]
            x, y = int(landmark.x * width), int(landmark.y * height)
            points.append((x, y))
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding
        padding_x = int((x_max - x_min) * 0.2)
        padding_y = int((y_max - y_min) * 0.2)
        
        x_min = max(0, x_min - padding_x)
        x_max = min(width, x_max + padding_x)
        y_min = max(0, y_min - padding_y)
        y_max = min(height, y_max + padding_y)
        
        roi = image_array[y_min:y_max, x_min:x_max]
        
        return Image.fromarray(roi), Image.fromarray(cv2.rectangle(image_array.copy(), 
                                                                 (x_min, y_min), 
                                                                 (x_max, y_max), 
                                                                 (0, 255, 0), 2))
    except Exception as e:
        st.error(f"Error during detection: {str(e)}")
        return None, None

def load_model():
    try:
        model_path = 'models/final_anemia_model.keras'
        if not os.path.exists(model_path):
            st.error("Model file not found")
            return None
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    if not isinstance(image, Image.Image):
        img = Image.open(image)
    else:
        img = image
    img = img.resize((160, 160))
    img = img.convert('RGB')
    img_array = img_to_array(img)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_anemia(model, image):
    img_processed = preprocess_image(image)
    prediction = model.predict(img_processed)
    confidence = abs(prediction[0][0] - 0.5) * 2
    return prediction[0][0] > 0.5, confidence

st.title('Anemia Detection System')
st.write('A medical screening tool that analyzes conjunctival images for potential anemia indicators.')

model = load_model()

uploaded_file = st.file_uploader("Upload Eye Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    conjunctiva_region, detection_vis = detect_conjunctiva(image)
    
    if conjunctiva_region is None:
        st.error("Detection failed. Please ensure the eye is clearly visible in the image.")
    else:
        st.subheader("Image Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.image(detection_vis, caption='Region of Interest', use_container_width=True)
        with col2:
            st.image(conjunctiva_region, caption='Processed Region', use_container_width=True)
        
        st.session_state.conjunctiva_region = conjunctiva_region
        
        if st.button("Analyze Image"):
            st.session_state.prediction_made = True

        if st.session_state.prediction_made:
            try:
                with st.spinner('Processing image...'):
                    is_anemic, confidence = predict_anemia(model, st.session_state.conjunctiva_region)
                    
                    st.subheader('Analysis Results')
                    
                    if is_anemic:
                        st.error(f'Potential anemia detected (confidence: {confidence:.1%})')
                    else:
                        st.success(f'No indication of anemia (confidence: {confidence:.1%})')
                    
                    st.warning('This is a screening tool only and should not replace professional medical diagnosis.')
            except Exception as e:
                st.error(f'Error during analysis: {str(e)}')
                st.session_state.prediction_made = False

st.markdown("""
### Usage Instructions
1. Take a clear photograph showing your inner lower eyelid (conjunctiva)
2. Pull down lower eyelid to expose inner surface
3. Ensure good lighting and steady positioning
4. Keep eye open and minimize reflections/shadows
""")
