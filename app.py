import streamlit as st
st.set_page_config(page_title="Anemia Detection", layout="wide")

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

def detect_conjunctiva(image):
    try:
        image = image.convert('RGB')
        image_array = np.array(image)
        height, width = image_array.shape[:2]
        
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        
        # Target central conjunctiva color
        lower_red = np.array([0, 50, 160])
        upper_red = np.array([10, 200, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        
        # Focus on center region
        y_min = int(height * 0.35)
        y_max = int(height * 0.65)
        x_min = int(width * 0.3)
        x_max = int(width * 0.7)
        
        mask[:y_min, :] = 0
        mask[y_max:, :] = 0
        mask[:, :x_min] = 0
        mask[:, x_max:] = 0
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
            
        # Get center-most contour
        center_y = height // 2
        center_contour = min(contours, key=lambda c: abs(cv2.boundingRect(c)[1] - center_y))
        x, y, w, h = cv2.boundingRect(center_contour)
        
        # Tight crop
        x = max(x_min, x - w//20)
        w = min(x_max - x, int(w * 1.1))
        y = max(y_min, y - h//20)
        h = min(y_max - y, int(h * 1.1))
        
        return (Image.fromarray(image_array[y:y+h, x:x+w]), 
                Image.fromarray(cv2.rectangle(image_array.copy(), (x,y), (x+w,y+h), (0,255,0), 2)))
    except Exception as e:
        st.write("Error:", str(e))
        return None, None
       
def load_model():
   return tf.keras.models.load_model('models/final_anemia_model.keras')

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
                       st.error(f'Potential anemia detected with {confidence:.1%} confidence')
                   else:
                       st.success(f'No indication of anemia (confidence: {confidence:.1%})')
                   
                   st.warning('This is a screening tool only and should not replace professional medical diagnosis.')
           except Exception as e:
               st.error(f'An error occurred during analysis: {str(e)}')
               st.session_state.prediction_made = False

st.subheader('Usage Instructions')
st.write('1. Capture a clear photograph of the lower inner eyelid')
st.write('2. Verify the detected region in the preview')
st.write('3. Proceed with analysis')

st.subheader('Image Quality Guidelines')
st.write('- Use consistent, adequate lighting')
st.write('- Ensure clear visibility of the inner eyelid surface')
st.write('- Maintain steady positioning during capture')
st.write('- Minimize reflections and shadows')
