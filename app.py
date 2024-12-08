import streamlit as st
st.set_page_config(page_title="Anemia Detection", layout="wide")

import os
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.preprocessing.image import img_to_array
import requests
import cv2

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

def create_curved_mask(image, pred, class_name):
    """Create a crescent-shaped mask with higher center point"""
    try:
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Get bbox center points with adjusted dimensions
        x = max(0, int(pred['x'] - pred['width']/2))
        y = max(0, int(pred['y'] - pred['height']/2))
        w = min(width - x, int(pred['width'] * 0.9))  # Keep width reduction
        h = min(height - y, int(pred['height'] * 1.5))  # Keep height increase
        
        if w <= 0 or h <= 0:
            return None, None
            
        # Create points for the crescent shape
        num_points = 150
        x_points = np.linspace(x, x + w, num_points)
        
        # Move center point even higher
        center_y = y + h/2.5  # Changed from 2.2 to 2.5 for higher center
        amplitude = h/2.6  # Keep same amplitude
        
        # Create curves
        angle = np.pi * (x_points - x) / w
        sin_values = np.sin(angle)
        sin_values = np.clip(sin_values, 0, 1)
        
        # Keep same curve proportions
        upper_curve = center_y + amplitude * 1.3 * sin_values
        lower_curve = center_y + (amplitude * 0.7) * sin_values
        
        # Keep same tapering
        taper = np.power(sin_values, 0.4)
        
        # Apply tapering
        curve_diff = upper_curve - lower_curve
        upper_curve = lower_curve + curve_diff * taper
        
        # Create final points
        points = np.vstack([
            np.column_stack([x_points, upper_curve]),
            np.column_stack([x_points[::-1], lower_curve[::-1]])
        ])
        
        # Ensure points stay within image bounds
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        
        # Convert to proper format for drawing
        polygon_points = points.astype(np.float32)
        
        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_points.astype(np.int32)], 255)
        
        return mask, polygon_points
        
    except Exception as e:
        st.error(f"Error creating curved mask: {str(e)}")
        return None, None
        
def detect_conjunctiva(image):
    try:
        # Basic preprocessing
        processed_image = image
        if image.mode == 'RGBA':
            processed_image = image.convert('RGB')
        
        # Save image temporarily
        temp_path = "temp_image.jpg"
        processed_image.save(temp_path)
        
        with open(temp_path, "rb") as image_file:
            image_data = image_file.read()
        
        api_url = f"{detector_model['api_url']}/eye-conjunctiva-detector/2"
        
        # Make prediction request
        response = requests.post(
            api_url,
            params={
                "api_key": detector_model['api_key'],
                "confidence": 30,
                "overlap": 50
            },
            files={"file": ("image.jpg", open(temp_path, "rb"), "image/jpeg")}
        )
        
        # Remove temp file
        os.remove(temp_path)
        
        if response.status_code != 200:
            st.error("Error connecting to detection service")
            return None, None, None
            
        predictions = response.json()
        
        if not predictions.get('predictions'):
            return None, None, None
            
        # Get the prediction with highest confidence
        pred = max(predictions['predictions'], key=lambda x: x['confidence'])
        class_name = pred['class']
        
        # Create curved mask
        mask, polygon_points = create_curved_mask(processed_image, pred, class_name)
        
        if mask is not None and polygon_points is not None:
            # Create RGBA version for transparent background
            img_array = np.array(processed_image)
            rgba = cv2.cvtColor(img_array, cv2.COLOR_RGB2RGBA)
            
            # Apply mask
            rgba[mask == 0] = [0, 0, 0, 0]
            
            # Find bounds of non-zero (non-transparent) region
            coords = cv2.findNonZero(mask)
            x, y, w, h = cv2.boundingRect(coords)
            
            # Extract conjunctiva region with transparency
            conjunctiva_region = rgba[y:y+h, x:x+w]
            
            # Create visualization with curved outline
            vis_image = processed_image.copy()
            vis_array = np.array(vis_image)
            
            # Draw filled polygon with transparency
            overlay = vis_array.copy()
            cv2.fillPoly(overlay, [polygon_points.astype(np.int32)], (0, 255, 0))
            alpha = 0.3
            vis_array = cv2.addWeighted(overlay, alpha, vis_array, 1 - alpha, 0)
            
            # Draw outline
            cv2.polylines(vis_array, [polygon_points.astype(np.int32)], True, (0, 255, 0), 2)
            
            return Image.fromarray(conjunctiva_region), Image.fromarray(vis_array), pred['confidence']
        
        return None, None, None
        
    except Exception as e:
        st.error("Error during image processing")
        st.write("Error details:", str(e))
        return None, None, None

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

uploaded_file = st.file_uploader("Upload Eye Image", type=['jpg', 'jpeg', 'png'], key="eye_image_upload")

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
        
        if st.button("Analyze for Anemia", key="analyze_button"):
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
