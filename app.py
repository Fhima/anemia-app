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
   """Create a curved mask following the conjunctiva contour"""
   try:
       # Convert to numpy array
       img_array = np.array(image)
       height, width = img_array.shape[:2]
       
       # Get bbox center points
       x = int(pred['x'] - pred['width']/2)
       y = int(pred['y'] - pred['height']/2)
       w = int(pred['width'])
       h = int(pred['height'])
       
       # Adjust curve parameters based on class
       if class_name == 'forniceal_palpebral':
           curve_height = h/3
       elif class_name == 'palpebral':
           curve_height = h/4
       else:
           curve_height = h/3.5
       
       # Create points for curve fitting
       x_points = np.linspace(x, x + w, num=50)
       # Create curved line following conjunctiva shape
       y_curve = y + h/2 + curve_height * np.sin(np.pi * (x_points - x) / w)
       
       # Create top curve with less curvature
       y_top = y + h/2 - curve_height * np.sin(np.pi * (x_points - x) / w) * 0.7
       
       # Combine points for complete shape
       curve_points = np.column_stack([x_points, y_curve])
       top_points = np.column_stack([x_points[::-1], y_top[::-1]])
       polygon_points = np.vstack([curve_points, top_points])
       
       # Create mask
       mask = np.zeros((height, width), dtype=np.uint8)
       cv2.fillPoly(mask, [polygon_points.astype(np.int32)], 1)
       
       return mask, polygon_points
   except Exception as e:
       st.error(f"Error creating curved mask: {str(e)}")
       return None, None

def preprocess_for_detection(image):
   """Preprocess image for detection"""
   try:
       # Convert to RGB if needed
       if image.mode == 'RGBA':
           image = image.convert('RGB')
           
       # Basic preprocessing
       width, height = image.size
       crop_height = int(height * 0.6)
       crop_top = int(height * 0.2)
       cropped = image.crop((0, crop_top, width, crop_top + crop_height))
       
       return cropped
   except Exception as e:
       st.error(f"Error preprocessing image: {str(e)}")
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
           # Apply mask to image
           img_array = np.array(processed_image)
           masked_image = img_array.copy()
           for c in range(3):
               masked_image[:,:,c] = img_array[:,:,c] * mask
           
           # Find masked region bounds
           coords = cv2.findNonZero(mask)
           x, y, w, h = cv2.boundingRect(coords)
           conjunctiva_region = masked_image[y:y+h, x:x+w]
           
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
