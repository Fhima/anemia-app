import streamlit as st
st.set_page_config(page_title="Anemia Detection", layout="wide")

import os
import tensorflow as tf
import numpy as np
from PIL import Image
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
        
        # Make prediction request
        response = requests.post(
            f"{detector_model['api_url']}/eye-conjunctiva-detector/2",
            params={
                "api_key": detector_model['api_key'],
                "confidence": 30,
            },
            files={"file": ("image.jpg", open(temp_path, "rb"), "image/jpeg")}
        )
        
        os.remove(temp_path)
        
        if response.status_code != 200:
            st.error("Error connecting to detection service")
            return None, None, None
            
        predictions = response.json()
        st.write("API Response:", predictions)  # Debug output
        
        if not predictions.get('predictions'):
            return None, None, None
            
        return image, image, 0.5  # Simple return for testing
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, None, None

# Simple UI for testing
st.title('Anemia Detection System - Test Version')

uploaded_file = st.file_uploader("Upload Eye Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    result = detect_conjunctiva(image)
    if result[0] is not None:
        st.image(image, caption="Uploaded Image")
       
def preprocess_for_anemia_detection(image):
   """Preprocess ROI exactly as done in training"""
   try:
       # Ensure image is RGB
       if image.mode != 'RGB':
           image = image.convert('RGB')
           
       # Resize to exact training dimensions
       image = image.resize((160, 160))
       
       # Convert to array and preprocess as in training
       img_array = img_to_array(image)
       img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
       
       # Add batch dimension
       img_array = np.expand_dims(img_array, axis=0)
       
       return img_array
   except Exception as e:
       st.error(f"Error in preprocessing: {str(e)}")
       return None

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

def predict_anemia(model, image):
   try:
       # Preprocess exactly as training
       img_array = preprocess_for_anemia_detection(image)
       if img_array is None:
           return None, None
           
       # Get prediction
       prediction = model.predict(img_array)
       
       # Apply class weights in inference
       anemic_prob = prediction[0][0] * 0.9  # Anemic class weight
       non_anemic_prob = (1 - prediction[0][0]) * 1.2  # Non-anemic class weight
       
       # Normalize probabilities
       total = anemic_prob + non_anemic_prob
       anemic_prob = anemic_prob / total
       
       # Calculate confidence
       confidence = abs(anemic_prob - 0.5) * 2
       
       return anemic_prob > 0.5, confidence
   except Exception as e:
       st.error(f"Error in prediction: {str(e)}")
       return None, None

# App UI
st.title('Anemia Detection System')
st.write('A medical screening tool that analyzes conjunctival images for potential anemia indicators.')

with st.container():
   st.markdown("""
   ### Usage Instructions
   1. Take a clear photograph focusing specifically on the lower eyelid area:
      - Pull down the lower eyelid to clearly expose the inner surface
      - Frame the photo to show mainly the conjunctiva (inner red/pink area)
      - Minimize the amount of surrounding eye area in the frame
   2. Ensure proper lighting:
      - Use consistent, even lighting
      - Avoid harsh shadows or reflections
   3. Keep the eye steady and in focus
   4. The photo should be similar to medical reference images of conjunctiva examinations
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
