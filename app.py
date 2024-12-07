import streamlit as st
st.set_page_config(page_title="Anemia Detection", layout="wide")

import os
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
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

def extract_curved_conjunctiva(image):
   """Extract just the curved conjunctiva region using a mask"""
   try:
       # Convert to array for processing
       img_array = np.array(image)
       
       # Create a mask in the shape of the conjunctiva (curved/smile shape)
       height, width = img_array.shape[:2]
       mask = np.zeros((height, width), dtype=np.uint8)
       
       # Calculate control points for curve
       center_x = width // 2
       center_y = height // 2
       curve_height = height // 4
       
       # Create curved mask using numpy operations
       x = np.arange(width)
       y = np.arange(height)
       X, Y = np.meshgrid(x, y)
       
       # Create a smile-shaped curve
       curve = center_y + curve_height * np.sin(np.pi * (X - center_x) / width)
       mask = (Y > curve - curve_height/2) & (Y < curve + curve_height/2)
       
       # Apply mask to image
       result = img_array.copy()
       for c in range(3):  # Apply to each color channel
           result[:,:,c] = img_array[:,:,c] * mask
           
       # Convert back to PIL Image
       masked_image = Image.fromarray(result)
       
       # Crop to non-zero region
       bbox = masked_image.getbbox()
       if bbox:
           masked_image = masked_image.crop(bbox)
           
       return masked_image
   except Exception as e:
       st.error(f"Error in curved extraction: {str(e)}")
       return image

def preprocess_for_detection(image):
    """Preprocess image to better match the training data format"""
    try:
        # Convert to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')
            
        # Do a gentler crop focusing on the lower eyelid
        width, height = image.size
        # Adjust these values to get a better crop of just the conjunctiva
        crop_height = int(height * 0.3)  # Smaller crop height
        crop_top = int(height * 0.4)     # Start a bit lower
        
        # Basic rectangular crop
        cropped = image.crop((0, crop_top, width, crop_top + crop_height))
        
        # Enhance contrast slightly
        enhancer = ImageEnhance.Contrast(cropped)
        enhanced = enhancer.enhance(1.1)  # Reduced from 1.2 to be more subtle
        
        return enhanced
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return image

def standardize_conjunctiva_image(image):
   """Standardize the cropped conjunctiva image to match CP-AnemicC format"""
   try:
       # Convert to RGB if needed
       if isinstance(image, np.ndarray):
           image = Image.fromarray(image)
       if image.mode != 'RGB':
           image = image.convert('RGB')
           
       # Standardize size while maintaining aspect ratio
       target_width = 160  # Match your model's input size
       aspect_ratio = image.width / image.height
       target_height = int(target_width / aspect_ratio)
       image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
       
       # Add padding if needed to match square input
       if target_height != target_width:
           new_img = Image.new('RGB', (target_width, target_width), (255, 255, 255))
           paste_y = (target_width - target_height) // 2
           new_img.paste(image, (0, paste_y))
           image = new_img
           
       # Convert to numpy array
       img_array = np.array(image)
       
       # Normalize colors
       img_array = img_array.astype(np.float32)
       img_array = img_array / 255.0
       
       return Image.fromarray((img_array * 255).astype(np.uint8))
   except Exception as e:
       st.error(f"Error standardizing image: {str(e)}")
       return image

def detect_conjunctiva(image):
   try:
       # Preprocess image
       processed_image = preprocess_for_detection(image)
       
       # Save image temporarily
       temp_path = "temp_image.jpg"
       processed_image.save(temp_path)
       
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
           return None, None, None
           
       predictions = response.json()
       
       if not predictions.get('predictions'):
           return None, None, None
           
       # Get the prediction with highest confidence
       pred = max(predictions['predictions'], key=lambda x: x['confidence'])
       
       # Extract bbox
       x = int(pred['x'] - pred['width']/2)
       y = int(pred['y'] - pred['height']/2)
       w = int(pred['width'])
       h = int(pred['height'])
       
       # Ensure coordinates are within image bounds
       image_array = np.array(processed_image)
       height, width = image_array.shape[:2]
       x = max(0, x)
       y = max(0, y)
       w = min(width - x, w)
       h = min(height - y, h)
       
       # Extract region
       conjunctiva_region = image_array[y:y+h, x:x+w]
       
       # Create visualization
       vis_image = processed_image.copy()
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
   try:
       # Standardize the conjunctiva image
       standardized_img = standardize_conjunctiva_image(image)
       
       # Show standardized image (for debugging)
       st.image(standardized_img, caption="Standardized Image for Analysis", width=200)
       
       # Preprocess for model
       img_processed = preprocess_image(standardized_img)
       
       # Get prediction
       prediction = model.predict(img_processed)
       confidence = abs(prediction[0][0] - 0.5) * 2
       
       return prediction[0][0] > 0.5, confidence
   except Exception as e:
       st.error(f"Error in anemia prediction: {str(e)}")
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
