import streamlit as st
st.set_page_config(page_title="Anemia Detection", layout="wide")

import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import mediapipe as mp

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'conjunctiva_region' not in st.session_state:
    st.session_state.conjunctiva_region = None

@st.cache_resource
def load_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

face_mesh = load_face_mesh()

# Keep your original functions here
def detect_conjunctiva(image):
    image = image.convert('RGB')
    image_array = np.array(image)
    height, width = image_array.shape[:2]
    
    results = face_mesh.process(image_array)
    
    if not results.multi_face_landmarks:
        return None, None
        
    face_landmarks = results.multi_face_landmarks[0]
    lower_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133]
    
    lower_eye_points = []
    for idx in lower_eye_indices:
        point = face_landmarks.landmark[idx]
        x, y = int(point.x * width), int(point.y * height)
        lower_eye_points.append((x, y))
    
    x_coords = [p[0] for p in lower_eye_points]
    y_coords = [p[1] for p in lower_eye_points]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    padding = int(width * 0.05)
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width, x_max + padding)
    y_max = min(height, y_max + padding)
    
    conjunctiva_region = image_array[y_min:y_max, x_min:x_max]
    vis_image = image_array.copy()
    cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    return Image.fromarray(conjunctiva_region), Image.fromarray(vis_image)

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

# Basic app layout
st.title('Anemia Detection System')
st.write('A medical screening tool that analyzes conjunctival images for potential anemia indicators.')

# Load model
model = load_model()

# Upload section
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

# Instructions
st.subheader('Usage Instructions')
st.write('1. Capture a clear photograph of the lower inner eyelid')
st.write('2. Verify the detected region in the preview')
st.write('3. Proceed with analysis')

st.subheader('Image Quality Guidelines')
st.write('- Use consistent, adequate lighting')
st.write('- Ensure clear visibility of the inner eyelid surface')
st.write('- Maintain steady positioning during capture')
st.write('- Minimize reflections and shadows')
</div>
""", unsafe_allow_html=True)
