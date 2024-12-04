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

# Your existing functions here...
[previous functions for detect_conjunctiva, load_model, preprocess_image, predict_anemia]

# Simple styling
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
st.write("""
1. Capture a clear photograph of the lower inner eyelid
2. Verify the detected region in the preview
3. Proceed with analysis
""")

st.subheader('Image Quality Guidelines')
st.write("""
- Use consistent, adequate lighting
- Ensure clear visibility of the inner eyelid surface
- Maintain steady positioning during capture
- Minimize reflections and shadows
""")
</ul>
</div>
""", unsafe_allow_html=True)
