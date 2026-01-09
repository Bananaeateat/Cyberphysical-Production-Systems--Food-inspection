"""
Prediction Page - Upload and classify food images
"""

import streamlit as st
from PIL import Image
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import load_model
from utils.image_processor import preprocess_image, validate_image, get_image_info
from utils.prediction import predict_food_quality, get_prediction_color, get_prediction_emoji, interpret_confidence

# Page configuration
st.set_page_config(
    page_title="Food Quality Prediction",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 Food Quality Prediction")
st.markdown("Upload a food image to check if it's fresh or stale")
st.markdown("---")

# Load model
model, error = load_model()

# Check if model is loaded
if error:
    st.error(f"⚠️ {error}")
    st.info("""
    ### How to fix this:
    1. Navigate to the project directory in terminal
    2. Run: `python train.py`
    3. Wait for training to complete (approximately 10 epochs)
    4. Refresh this page

    The model will be saved to `models/food_quality_detector.h5`
    """)
    st.stop()

st.success("✅ Model loaded successfully!")
st.markdown("---")

# File uploader
st.subheader("📤 Upload Image")
uploaded_file = st.file_uploader(
    "Choose a food image (JPG, JPEG, PNG, BMP)",
    type=['jpg', 'jpeg', 'png', 'bmp'],
    help="Upload an image of food to classify as fresh or stale"
)

if uploaded_file is not None:
    # Validate file
    if not validate_image(uploaded_file):
        st.error("❌ Invalid file format. Please upload JPG, JPEG, PNG, or BMP images.")
        st.stop()

    try:
        # Load image
        image = Image.open(uploaded_file)

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Display image and prediction side by side
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(image, use_container_width=True)

            # Show image info in expander
            with st.expander("ℹ️ Image Information"):
                img_info = get_image_info(image)
                st.write(f"**Original Size**: {img_info['width']} x {img_info['height']} pixels")
                st.write(f"**Color Mode**: {img_info['mode']}")
                st.write(f"**File Name**: {uploaded_file.name}")
                st.write(f"**File Size**: {uploaded_file.size / 1024:.2f} KB")

        with col2:
            st.subheader("🎯 Prediction Result")

            # Show processing message
            with st.spinner("Processing image..."):
                # Preprocess image
                preprocessed_img = preprocess_image(image)

                # Make prediction
                result = predict_food_quality(model, preprocessed_img)

            # Display result
            label = result['label']
            confidence = result['confidence']
            raw_pred = result['raw_prediction']

            # Get color and emoji
            color = get_prediction_color(label)
            emoji = get_prediction_emoji(label)

            # Display large result
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: {color}22; border-radius: 10px; border: 3px solid {color};">
                <h1 style="color: {color}; font-size: 48px; margin: 0;">{emoji} {label}</h1>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Confidence display
            st.markdown(f"### Confidence: {confidence:.2f}%")
            st.progress(confidence / 100)

            # Confidence interpretation
            confidence_level = interpret_confidence(confidence)
            if confidence >= 75:
                st.success(f"Confidence Level: **{confidence_level}**")
            elif confidence >= 60:
                st.info(f"Confidence Level: **{confidence_level}**")
            else:
                st.warning(f"Confidence Level: **{confidence_level}** - Consider retaking the image")

            # Technical details in expander
            with st.expander("🔬 Technical Details"):
                st.write("**Model Information:**")
                st.write("- Architecture: MobileNetV2 (Transfer Learning)")
                st.write("- Input Size: 224 x 224 pixels")
                st.write("- Output: Binary Classification (Sigmoid)")
                st.write("")
                st.write("**Prediction Details:**")
                st.write(f"- Raw Output Value: {raw_pred:.6f}")
                st.write(f"- Threshold: 0.5")
                st.write(f"- Classification: {'Fresh (< 0.5)' if raw_pred < 0.5 else 'Stale (≥ 0.5)'}")
                st.write(f"- Confidence Score: {confidence:.2f}%")

        # Interpretation
        st.markdown("---")
        st.subheader("📖 Interpretation")

        if label == "Fresh":
            st.success("""
            ✅ **Fresh Food Detected**

            The model predicts this food item is **fresh** and safe to consume.
            The neural network has identified visual characteristics typical of fresh food,
            such as vibrant colors, firm texture, and absence of decay signs.
            """)
        else:
            st.error("""
            ⚠️ **Stale Food Detected**

            The model predicts this food item is **stale** and may not be safe to consume.
            The neural network has identified visual characteristics typical of spoiled food,
            such as discoloration, soft spots, mold, or other signs of decay.

            **Recommendation**: Do not consume this food item.
            """)

        # Additional info
        st.info("""
        **Note**: This is an AI prediction based on visual features. Always use your own judgment
        and follow food safety guidelines. When in doubt, it's better to discard questionable food items.
        """)

    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.info("Please try uploading a different image or check the image format.")

else:
    # Instructions when no file is uploaded
    st.info("""
    ### 📋 Instructions:

    1. Click the **Browse files** button above
    2. Select a food image from your device
    3. Supported formats: JPG, JPEG, PNG, BMP
    4. Wait for the prediction result
    5. View the classification and confidence score

    ### 💡 Tips for Best Results:

    - Use clear, well-lit images
    - Ensure the food item is in focus
    - Avoid images with heavy filters or edits
    - Single food items work best
    - Close-up shots are recommended

    ### 🎯 What the Model Detects:

    The model classifies food as:
    - **Fresh** (🍎): Food that appears safe to eat, with good color and texture
    - **Stale** (🤢): Food showing signs of spoilage, discoloration, or decay
    """)

    # Example predictions section
    st.markdown("---")
    st.subheader("📸 Example Images")
    st.write("Don't have an image? Try the **Sample Gallery** page to see the model in action!")

st.markdown("---")
st.caption("Food Quality Detection System | Powered by MobileNetV2 & Streamlit")
