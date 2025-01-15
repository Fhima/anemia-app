import streamlit as st
st.set_page_config(page_title="Anemia Detection", layout="wide")

import os
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.preprocessing.image import img_to_array
import requests
import cv2
from config import Config

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'conjunctiva_region' not in st.session_state:
    st.session_state.conjunctiva_region = None

# Initialize Roboflow
@st.cache_resource
def load_roboflow():
    Config.validate_config()
    return {
        "api_url": Config.ROBOFLOW_API_URL,
        "api_key": Config.ROBOFLOW_API_KEY
    }

detector_model = load_roboflow()

def create_curved_mask(image, pred, class_name):
    """Create an upturned crescent-shaped mask for conjunctiva"""
    try:
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Get bbox center points with moderate upward shift
        x = max(0, int(pred['x'] - pred['width']/2))
        y = max(0, int(pred['y'] - pred['height']/2)) - int(pred['height']/5)  # Consistent upward shift
        
        # Keep working proportions
        w = min(width - x, int(pred['width'] * 1.1))
        h = min(height - y, int(pred['height'] * 1.4))
        
        if w <= 0 or h <= 0:
            return None, None
            
        # Create points for the crescent shape
        num_points = 150
        x_points = np.linspace(x, x + w, num_points)
        
        # Keep successful parameters from before
        center_y = y + h/4.2
        amplitude = h/2.4
        
        # Create curves with restored orientation
        angle = np.pi * (x_points - x) / w
        sin_values = np.sin(angle)
        sin_values = np.clip(sin_values, 0, 1)
        
        # Return to successful curve proportions
        upper_curve = center_y + amplitude * 1.5 * sin_values  # Upper curve
        lower_curve = center_y + (amplitude * 0.6) * sin_values  # Lower curve
        
        # Keep successful tapering
        taper = np.power(sin_values, 0.45)
        
        # Apply tapering
        curve_diff = upper_curve - lower_curve
        upper_curve = lower_curve + curve_diff * taper
        
        # Create final points with original orientation
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
    """Preprocess ROI to match training data augmentation parameters"""
    try:
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Convert to array
        img_array = np.array(image)
        
        # Color adjustments matching training augmentation
        # RandomBrightness(0.12) equivalent
        brightness_factor = 1.12  # Slight increase to match training
        img_array = tf.image.adjust_brightness(img_array, delta=0.12)
        
        # RandomContrast(0.18) equivalent
        img_array = tf.image.adjust_contrast(img_array, contrast_factor=1.18)
        
        # Maintain consistent size (160x160)
        image = tf.image.resize(img_array, (160, 160), method='bilinear')
        
        # Convert to float32 and apply EfficientNet preprocessing
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
        
        return np.expand_dims(image, axis=0)
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None

@st.cache_resource
def load_model():
    """Download and load the model from Google Drive"""
    try:
        model_dir = 'models'
        model_path = os.path.join(model_dir, 'final_anemia_model.keras')
        
        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Download model if needed
        if not os.path.exists(model_path):
            import gdown
            
            with st.spinner('Loading model resources...'):
                url = "https://drive.google.com/file/d/1_0laYs2WeMqeDqaPPHmYzgUtxoKLZNfG/view?usp=sharing"
                output = gdown.download(url, output=model_path, quiet=True, fuzzy=True)
                
                if not output or not os.path.exists(model_path):
                    st.error("Error loading model resources")
                    return None
        
        return tf.keras.models.load_model(model_path)
            
    except Exception as e:
        st.error("Error loading model resources")
        return None

def predict_anemia(model, image):
    """Predict anemia with class weights and threshold"""
    try:
        # Preprocess
        img_array = preprocess_for_anemia_detection(image)
        if img_array is None:
            return None, None
            
        # Get model prediction (single value between 0-1)
        pred = model.predict(img_array)[0][0]
        
        # Apply weights and calculate normalized probability
        weighted_ratio = (pred * 1.2) / ((pred * 1.2) + ((1 - pred) * 0.9))
        
        # Classify with threshold and return confidence
        is_anemic = weighted_ratio < 0.45  # Equivalent to anemic_prob > 0.85
        confidence = max(weighted_ratio, 1 - weighted_ratio)
        
        return is_anemic, confidence
        
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
