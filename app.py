import streamlit as st
st.set_page_config(page_title="Anemia Detection", layout="wide")

import os
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from inference_sdk import InferenceHTTPClient

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'conjunctiva_region' not in st.session_state:
    st.session_state.conjunctiva_region = None

# Initialize Roboflow
@st.cache_resource
def load_roboflow():
    return InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="g6W2V0dcNuMVTkygIv9G"
    )

detector_model = load_roboflow()

def draw_box(image, box, color=(0, 255, 0)):
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    x, y, w, h = box
    
    # Draw rectangle using numpy operations
    img_array[y:y+2, x:x+w] = color  # Top line
    img_array[y+h-2:y+h, x:x+w] = color  # Bottom line
    img_array[y:y+h, x:x+2] = color  # Left line
    img_array[y:y+h, x+w-2:x+w] = color  # Right line
    
    return Image.fromarray(img_array)

def detect_conjunctiva(image):
    try:
        # Save image temporarily
        temp_path = "temp_image.jpg"
        image.save(temp_path)
        
        # Get prediction using inference client
        prediction = detector_model.infer(temp_path, model_id="eye-conjunctiva-detector/2")
        
        # Remove temp file
        os.remove(temp_path)
        
        if not prediction:
            return None, None
            
        # Get the prediction with highest confidence
        pred = max(prediction, key=lambda x: x['confidence'])
        
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
        
        # Create visualization using PIL instead of cv2
        vis_image = draw_box(image, (x, y, w, h))
        
        return Image.fromarray(conjunctiva_region), vis_image
        
    except Exception as e:
        st.error(f"Error in detection: {str(e)}")
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

# App UI
st.title('Anemia Detection System')
st.write('A medical screening tool that analyzes conjunctival images for potential anemia indicators.')

model = load_model()

uploaded_file = st.file_uploader("Upload Eye Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    conjunctiva_region, detection_vis = detect_conjunctiva(image)
    
    if conjunctiva_region is None:
        st.error("Could not detect conjunctiva. Please ensure the inner eyelid is clearly visible.")
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
