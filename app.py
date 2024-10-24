import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide"
)

# Load model at startup
@st.cache_resource
def load_model():
    try:
        # Define the model architecture exactly as in your training code
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)),
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
            tf.keras.layers.Dense(9, activation='softmax')  # 9 classes as in your code
        ])
        
        # Load weights
        model.load_weights('plant_disease1.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Class names and constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
class_names = [
    'Bacterial leaf blight', 'Brown spot', 'Gray_Leaf_Spot', 'Healthy',
    'Leaf smut', 'blight', 'common_rust', 'septoria', 'stripe_rust'
]

fertilizer_recommendations = {
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

def predict_image(image_path):
    """Prediction function using your notebook's logic"""
    img = tf.keras.preprocessing.image.load_img(
        image_path, 
        target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    model = load_model()
    if model is None:
        return None, None, None
        
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    fertilizer_advice = fertilizer_recommendations.get(
        predicted_class, 
        "No specific recommendation available."
    )
    
    return predicted_class, confidence, fertilizer_advice

def main():
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            margin-top: 1rem;
        }
        div[data-testid="stImage"] img {
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    # Header with attractive styling
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='color: #2e7d32;'>🌿 Plant Disease Classifier</h1>
            <p style='font-size: 1.2em; color: #666;'>
                Upload an image of a plant leaf to identify diseases and get fertilizer recommendations
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This application uses deep learning to detect plant diseases "
        "from leaf images and provides appropriate fertilizer recommendations."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        ### How to use:
        1. Upload a clear image of a plant leaf
        2. Wait for the analysis
        3. View results and recommendations
    """)

    # Main content in two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
            <div style=' padding: 20px; border-radius: 10px;'>
                <h3>Upload Image</h3>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with open("temp.jpg", "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("Analyze Leaf", key="analyze"):
                with st.spinner("Analyzing..."):
                    # Get prediction
                    predicted_class, confidence, fertilizer_advice = predict_image("temp.jpg")
                    
                    if predicted_class is not None:
                        # Display results in second column
                        with col2:
                            st.markdown("""
                                <div style=' padding: 20px; border-radius: 10px;'>
                                    <h3>Analysis Results</h3>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Disease prediction box
                            st.markdown(f"""
                                <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin: 20px 0;'>
                                    <h4 style='color: #2e7d32; margin: 0;'>Detected Disease:</h4>
                                    <h2 style='color: {'#2e7d32' if predicted_class == 'Healthy' else '#c62828'}; margin: 10px 0;'>
                                        {predicted_class}
                                    </h2>
                                    <p style='margin: 0;'>Confidence: {confidence:.2f}%</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Fertilizer recommendation box
                            st.markdown(f"""
                                <div style=' padding: 20px; border-radius: 10px; margin: 20px 0;'>
                                    <h4 style=' margin: 0;'>Fertilizer Recommendation:</h4>
                                    <p style='margin: 10px 0;'>{fertilizer_advice}</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Care instructions
                            st.markdown("""
                                <div style='background-color: #fff3e0; padding: 20px; border-radius: 10px;'>
                                    <h4 style='color: #e65100; margin: 0;'>Care Instructions:</h4>
                                    <ul style='margin: 10px 0;'>
                            """, unsafe_allow_html=True)
                            
                            if predicted_class != "Healthy":
                                st.markdown("""
                                    - Remove affected leaves to prevent spread
                                    - Ensure proper air circulation
                                    - Water at the base of the plant
                                    - Monitor for disease progression
                                """)
                            else:
                                st.markdown("""
                                    - Continue regular maintenance
                                    - Monitor water levels
                                    - Ensure adequate sunlight
                                    - Check for pest infestations
                                """)
                            
                            st.markdown("</ul></div>", unsafe_allow_html=True)
            
            # Clean up
            import os
            if os.path.exists("temp.jpg"):
                os.remove("temp.jpg")

if __name__ == "__main__":
    main()