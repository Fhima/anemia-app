import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'conjunctiva_region' not in st.session_state:
    st.session_state.conjunctiva_region = None

def detect_conjunctiva(image):
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

def load_model():
    return tf.keras.models.load_model('models/final_anemia_model.keras')

def preprocess_image(image):
    if not isinstance(image, Image.Image):
        img = Image.open(image)
    else:
        img = image
    img = img.resize((160, 160))  # Match your model's input size
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

st.set_page_config(page_title="Anemia Detection", layout="wide")

st.title('🔍 Anemia Detection System')
st.markdown("""
This system analyzes conjunctival images to detect potential anemia. Upload a clear photo of your eye's 
inner lower eyelid (conjunctiva) for analysis.
""")

# Load model
model = load_model()

# Image upload with custom styling
st.markdown("""
<style>
    .uploadedFile {
        border: 2px dashed #4e8df5;
        border-radius: 10px;
        padding: 20px;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Eye Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    conjunctiva_region, detection_vis = detect_conjunctiva(image)
    
    if conjunctiva_region is None:
        st.error("❌ Could not detect conjunctival region. Please ensure the inner eyelid is clearly visible.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.image(detection_vis, caption='Detected Region', use_container_width=True)
        with col2:
            st.image(conjunctiva_region, caption='Extracted Conjunctiva', use_container_width=True)
        
        st.session_state.conjunctiva_region = conjunctiva_region
        
        if st.button("✅ Confirm and Analyze", type="primary"):
            st.session_state.prediction_made = True

        if st.session_state.prediction_made:
            try:
                with st.spinner('🔄 Analyzing...'):
                    is_anemic, confidence = predict_anemia(model, st.session_state.conjunctiva_region)
                    
                    st.subheader('📊 Results')
                    if is_anemic:
                        st.error(f'⚠️ Potential Anemia Detected (Confidence: {confidence:.1%})')
                    else:
                        st.success(f'✅ No Anemia Detected (Confidence: {confidence:.1%})')
                    
                    st.warning('⚕️ Note: This is a screening tool only. Please consult a healthcare professional for proper diagnosis.')
            except Exception as e:
                st.error(f'❌ Error during analysis: {str(e)}')
                st.session_state.prediction_made = False

st.markdown("""
### 📋 Instructions
1. Take a clear photo showing your lower inner eyelid (conjunctiva)
2. Upload the image and verify the detected region
3. Click 'Confirm and Analyze' for results

### 📸 Tips for Good Images
- Ensure good lighting
- Pull down lower eyelid to expose inner surface
- Keep eye open and steady
- Avoid glare or shadows
""")