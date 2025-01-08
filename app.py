import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = [
    'Bacterial leaf blight', 'Brown spot', 'Gray_Leaf_Spot', 'Healthy',
    'Leaf smut', 'blight', 'common_rust', 'septoria', 'stripe_rust'
]

FERTILIZER_RECOMMENDATIONS = {
    'stripe_rust': 'Use a nitrogen-rich fertilizer (e.g., Urea) to promote growth.',
    'Healthy': 'Regular balanced fertilizer (e.g., NPK 10-10-10) is recommended.',
    'septoria': 'Use a potassium-rich fertilizer to enhance plant resilience.',
    'Brown spot': 'Apply a balanced fertilizer with micronutrients.',
    'Leaf smut': 'Use a nitrogen-phosphorus-potassium (NPK) fertilizer to boost recovery.',
    'Bacterial leaf blight': 'Use organic fertilizers like compost to improve soil health.',
    'Gray_Leaf_Spot': 'Apply a fertilizer high in potassium for improved disease resistance.',
    'common_rust': 'Use a balanced NPK fertilizer to maintain overall plant health.',
    'blight': 'Organic fertilizers and compost to enhance soil quality are recommended.',
}

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        # Create the model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create a dummy input to build the model
        dummy_input = tf.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3))
        model(dummy_input)
        
        # Load the weights
        try:
            model = tf.keras.models.load_model('plant_disease11.keras')
        except:
            try:
                model.load_weights('plant_disease11.keras')
            except:
                try:
                    model = tf.keras.models.load_model('plant_disease1.h5')
                except:
                    model.load_weights('plant_disease1.h5')
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess the image for model prediction"""
    img = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = tf.expand_dims(img_array, 0)
    return img_array

def predict_disease(image):
    """Predict disease from image"""
    model = load_model()
    if model is None:
        return None, None, None

    try:
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx] * 100)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        fertilizer_advice = FERTILIZER_RECOMMENDATIONS.get(predicted_class, "No specific recommendation available.")
        
        return predicted_class, confidence, fertilizer_advice
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

def display_prediction(predicted_class, confidence, fertilizer_advice):
    """Display prediction results with styling"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                <h4 style='color: #1f77b4; margin-bottom: 10px;'>Disease Detection Results</h4>
                <p><strong>Detected Condition:</strong> {}</p>
                <p><strong>Confidence:</strong> {:.2f}%</p>
            </div>
        """.format(predicted_class, confidence), unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                <h4 style='color: #2e7d32; margin-bottom: 10px;'>Fertilizer Recommendation</h4>
                <p>{}</p>
            </div>
        """.format(fertilizer_advice), unsafe_allow_html=True)

def main():
    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .uploadedFile {
            margin-bottom: 2rem;
        }
        .stButton>button {
            width: 100%;
            margin-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("🌿 Fertilizer and disease recommendation system")
    st.markdown("Upload a leaf image to detect diseases and get fertilizer recommendations")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict button
        if st.button("Analyze Leaf"):
            with st.spinner("Analyzing image..."):
                # Get prediction
                predicted_class, confidence, fertilizer_advice = predict_disease(image)
                
                if predicted_class:
                    # Display results
                    display_prediction(predicted_class, confidence, fertilizer_advice)
                    
                    # Additional care instructions
                    st.markdown("### 📋 Care Instructions")
                    if predicted_class != "Healthy":
                        st.warning("""
                            - Isolate affected plants to prevent disease spread
                            - Remove and dispose of infected leaves
                            - Improve air circulation around plants
                            - Apply recommended fertilizer as directed
                            - Monitor closely for disease progression
                        """)
                    else:
                        st.success("""
                            - Continue regular maintenance
                            - Water appropriately
                            - Ensure adequate sunlight
                            - Monitor for early signs of problems
                        """)

    # Sidebar information
    st.sidebar.title("About")
    st.sidebar.info("""
        This application uses deep learning to detect plant diseases from leaf images 
        and provides appropriate fertilizer recommendations. The model is trained to 
        recognize various common plant diseases and healthy plants.
    """)

if __name__ == "__main__":
    main()