import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import joblib
import cv2

# Session state
if 'prediction_made' not in st.session_state:
   st.session_state.prediction_made = False
if 'conjunctiva_region' not in st.session_state:
   st.session_state.conjunctiva_region = None

def detect_conjunctiva(image):
   """Detects and extracts conjunctival region"""
   image = image.convert('RGB')
   image_array = np.array(image)
   original = image_array.copy()
   
   hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
   lower_red = np.array([0, 30, 60])
   upper_red = np.array([20, 150, 255])
   
   mask = cv2.inRange(hsv, lower_red, upper_red)
   
   kernel = np.ones((5,5), np.uint8)
   mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
   mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
   
   contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
   if not contours:
       return None, None
   
   largest_contour = max(contours, key=cv2.contourArea)
   x, y, w, h = cv2.boundingRect(largest_contour)
   
   padding = 20
   y_start = max(0, y - padding)
   y_end = min(image_array.shape[0], y + h + padding)
   x_start = max(0, x - padding)
   x_end = min(image_array.shape[1], x + w + padding)
   
   conjunctiva_region = original[y_start:y_end, x_start:x_end]
   vis_image = original.copy()
   cv2.rectangle(vis_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
   
   return Image.fromarray(conjunctiva_region), Image.fromarray(vis_image)

def load_models():
   deploy_model = tf.keras.models.load_model('models/deploy_anemia_model.keras')
   scaler = joblib.load('models/clinical_scaler.joblib')
   return deploy_model, scaler

def preprocess_image(image):
    if not isinstance(image, Image.Image):
        img = Image.open(image)
    else:
        img = image
    img = img.resize((224, 224))
    img = img.convert('RGB')  # Convert to RGB
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_anemia(model, scaler, image, age_months, gender):
   img_processed = preprocess_image(image)
   age = np.array([[age_months]])
   gender_val = np.array([[1.0 if gender == 'Male' else 0.0]])
   prediction = model.predict([img_processed, age, gender_val])
   confidence = abs(prediction[0][0] - 0.5) * 2
   return prediction[0][0] > 0.5, confidence

# App
st.title('Anemia Detection System')

# Load models
model, scaler = load_models()

# Image upload
st.subheader('Upload Eye Image')
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

# Clinical inputs
col1, col2 = st.columns(2)
with col1:
   age_years = st.number_input('Age (Years)', min_value=0, max_value=100)
   age_months = age_years * 12
with col2:
   gender = st.selectbox('Gender', ['Male', 'Female'])

if uploaded_file:
   # Process image
   image = Image.open(uploaded_file)
   conjunctiva_region, detection_vis = detect_conjunctiva(image)
   
   if conjunctiva_region is None:
       st.error("Could not detect conjunctival region. Ensure inner eyelid is visible.")
   else:
       col1, col2 = st.columns(2)
       with col1:
           st.image(detection_vis, caption='Detected Region (Green Box)', use_column_width=True)
       with col2:
           st.image(conjunctiva_region, caption='Extracted Conjunctiva', use_column_width=True)
       
       # Store region in session state
       st.session_state.conjunctiva_region = conjunctiva_region
       
       if st.button("Confirm Detection and Proceed"):
           st.session_state.prediction_made = True

       if st.session_state.prediction_made:
           try:
               with st.spinner('Analyzing...'):
                   is_anemic, confidence = predict_anemia(
                       model, 
                       scaler, 
                       st.session_state.conjunctiva_region,
                       age_months,
                       gender
                   )
                   
                   st.subheader('Results')
                   if is_anemic:
                       st.error(f'Anemia Detected (Confidence: {confidence:.1%})')
                   else:
                       st.success(f'No Anemia Detected (Confidence: {confidence:.1%})')
                   
                   st.warning('Note: This is a screening tool and should not replace professional medical diagnosis.')
           except Exception as e:
               st.error(f'Error during prediction: {str(e)}')
               st.session_state.prediction_made = False

st.markdown("""
### Instructions:
1. Upload a photo showing your eye with the lower eyelid pulled down
2. Verify the system correctly identified the conjunctiva (inner eyelid)
3. Enter age in years
4. Select gender
5. Confirm detection and get results
6. Consult with a healthcare provider for proper diagnosis

**Note:** The system will automatically locate the conjunctiva. Make sure the inner eyelid is clearly visible.
""")