import streamlit as st
st.set_page_config(page_title="Anemia Detection", layout="wide")

import os
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.preprocessing.image import img_to_array
import requests

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'conjunctiva_region' not in st.session_state:
    st.session_state.conjunctiva_region = None

# Initialize Roboflow
@st.cache_resource
def load_roboflow():
    return {
        "api_url": "https://detect.roboflow.com",
        "api_key": "g6W2V0dcNuMVTkygIv9G"
    }

detector_model = load_roboflow()

def detect_conjunctiva(image):
    try:
        # Save image temporarily
        temp_path = "temp_image.jpg"
        image.save(temp_path)
        
        with open(temp_path, "rb") as image_file:
            image_data = image_file.read()
        
        api_url = f"{detector_model['api_url']}/eye-conjunctiva-detector/2"
        
        # Make prediction request
        response = requests.post(
            api_url,
            params={"api_key": detector_model['api_key']},
            files={"file": ("image.jpg", open(temp_path, "rb"), "image/jpeg")}
        )
        
        # Remove temp file
        os.remove(temp_path)
        
        if response.status_code != 200:
            st.error("Error connecting to detection service")
            return None, None
            
        predictions = response.json()
        
        if not predictions.get('predictions'):
            return None, None
            
        # Get the prediction with highest confidence
        pred = max(predictions['predictions'], key=lambda x: x['confidence'])
        
        # Extract bbox
        x = int(pred['x'] - pred['width']/2)
        y = int(pred['y'] - pred['height']/2)
        w = int(pred['width'])
        h = int(pred['height'])
        
        # Ensure coordinates are within image bounds
        image_array = np.array(image)
        height, width = image_array.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(width - x, w)
        h = min(height - y, h)
        
        # Extract region
        conjunctiva_region = image_array[y:y+h, x:x+w]
        
        # Create visualization
        vis_image = image.copy()
        draw = ImageDraw.Draw(vis_image)
        draw.rectangle([x, y, x+w, y+h], outline='green', width=3)
        
        return Image.fromarray(conjunctiva_region), vis_image, pred['confidence']
        
    except Exception as e:
        st.error("Error during image processing")
        return None, None, None

def load_model():
    try:
        model_path = 'models/final_anemia_model.keras'
        if not os.path.exists(model_path):
            st.error("Model file not found")
            return None
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error("Error loading anemia detection model")
        return None

def preprocess_image(image):
    img = image if isinstance(image, Image.Image) else Image.open(image)
    img = img.resize((160, 160))
    img = img.convert('RGB')
    img_array = img_to_array(img)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def predict_anemia(model, image):
    img_processed = preprocess_image(image)
    prediction = model.predict(img_processed)
    confidence = abs(prediction[0][0] - 0.5) * 2
    return prediction[0][0] > 0.5, confidence

# App UI
st.title('Anemia Detection System')
st.write('A medical screening tool that analyzes conjunctival images for potential anemia indicators.')

with st.container():
    st.markdown("""
    ### Usage Instructions
    1. Take a clear photograph showing your inner lower eyelid (conjunctiva)
    2. Pull down lower eyelid to expose inner surface
    3. Ensure good lighting and steady positioning
    4. Keep eye open and minimize reflections/shadows
    """)

model = load_model()

uploaded_file = st.file_uploader("Upload Eye Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    with st.spinner('Processing image...'):
        image = Image.open(uploaded_file)
        conjunctiva_region, detection_vis, confidence = detect_conjunctiva(image)
    
    if conjunctiva_region is None:
        st.error("Could not detect conjunctiva. Please ensure the inner eyelid is clearly visible.")
    else:
        st.success(f"Conjunctiva detected (Confidence: {confidence:.1%})")
        
        st.subheader("Image Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.image(detection_vis, caption='Region of Interest', use_container_width=True)
        with col2:
            st.image(conjunctiva_region, caption='Processed Region', use_container_width=True)
        
        st.session_state.conjunctiva_region = conjunctiva_region
        
        if st.button("Analyze for Anemia"):
            st.session_state.prediction_made = True

        if st.session_state.prediction_made:
            try:
                with st.spinner('Analyzing image...'):
                    is_anemic, confidence = predict_anemia(model, st.session_state.conjunctiva_region)
                    
                    st.subheader('Analysis Results')
                    if is_anemic:
                        st.error(f'Potential anemia detected (Confidence: {confidence:.1%})')
                    else:
                        st.success(f'No indication of anemia (Confidence: {confidence:.1%})')
                    
                    st.warning('This is a screening tool only and should not replace professional medical diagnosis.')
            except Exception as e:
                st.error('Error during analysis')
                st.session_state.prediction_made = False

st.markdown("---")
st.caption("Developed as a medical screening assistant. For research purposes only.")
