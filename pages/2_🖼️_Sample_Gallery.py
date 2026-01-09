"""
Sample Gallery Page - Test the model on sample images from the dataset
"""

import streamlit as st
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_loader import load_model
from utils.image_processor import preprocess_image, load_image_from_path
from utils.prediction import predict_food_quality, get_prediction_color, get_prediction_emoji
from utils.data_utils import get_sample_images

# Page configuration
st.set_page_config(
    page_title="Sample Gallery",
    page_icon="🖼️",
    layout="wide"
)

# Title
st.title("🖼️ Sample Gallery")
st.markdown("Test the model on sample images from the test dataset")
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
    3. Wait for training to complete
    4. Refresh this page
    """)
    st.stop()

# Controls
st.subheader("⚙️ Gallery Settings")

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    category = st.selectbox(
        "Select Category",
        options=['fresh', 'stale', 'both'],
        format_func=lambda x: x.capitalize(),
        help="Choose which category of images to display"
    )

with col2:
    num_samples = st.slider(
        "Number of Samples",
        min_value=1,
        max_value=20,
        value=6,
        help="Number of images to display (per category if 'both' selected)"
    )

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    refresh_button = st.button("🔄 Refresh", help="Load new random samples", use_container_width=True)

st.markdown("---")

# Get sample images
try:
    # Use session state to maintain samples unless refresh is clicked
    if 'samples' not in st.session_state or refresh_button:
        samples = get_sample_images(category, num_samples)
        st.session_state.samples = samples
    else:
        samples = st.session_state.samples

    if not samples:
        st.warning(f"""
        No images found in the test dataset for category: **{category}**

        Please ensure:
        1. The dataset is organized in `data/test/fresh/` and `data/test/stale/` directories
        2. Run `python organize_data.py` if you haven't already
        """)
        st.stop()

    # Display samples
    st.subheader(f"📸 Showing {len(samples)} Sample(s)")

    # Predict all button
    if st.button("🎯 Predict All", type="primary", use_container_width=False):
        st.session_state.predict_all = True

    st.markdown("---")

    # Display images in grid (5 per row)
    cols_per_row = 5
    correct_predictions = 0
    total_predictions = 0

    for idx in range(0, len(samples), cols_per_row):
        cols = st.columns(cols_per_row)

        for col_idx, col in enumerate(cols):
            sample_idx = idx + col_idx

            if sample_idx < len(samples):
                img_path, true_label = samples[sample_idx]

                with col:
                    try:
                        # Load and display image
                        image = load_image_from_path(img_path)
                        st.image(image, use_container_width=True)

                        # Show filename
                        filename = os.path.basename(img_path)
                        st.caption(f"📄 {filename[:20]}...")

                        # Show true label
                        st.write(f"**True**: {true_label}")

                        # Predict button or auto-predict
                        predict_key = f"predict_{sample_idx}"

                        if 'predict_all' in st.session_state and st.session_state.predict_all:
                            should_predict = True
                        else:
                            should_predict = st.button(
                                "Predict",
                                key=predict_key,
                                use_container_width=True
                            )

                        if should_predict:
                            # Preprocess and predict
                            with st.spinner("Predicting..."):
                                preprocessed = preprocess_image(image)
                                result = predict_food_quality(model, preprocessed)

                            label = result['label']
                            confidence = result['confidence']

                            # Display prediction
                            color = get_prediction_color(label)
                            emoji = get_prediction_emoji(label)

                            st.markdown(f"""
                            <div style="text-align: center; padding: 10px; background-color: {color}22;
                                 border-radius: 5px; border: 2px solid {color};">
                                <strong style="color: {color};">{emoji} {label}</strong>
                            </div>
                            """, unsafe_allow_html=True)

                            st.write(f"**Confidence**: {confidence:.1f}%")

                            # Check if correct
                            if label == true_label:
                                st.success("✓ Correct")
                                correct_predictions += 1
                            else:
                                st.error("✗ Incorrect")

                            total_predictions += 1

                    except Exception as e:
                        st.error(f"Error loading image: {str(e)}")

    # Reset predict_all flag
    if 'predict_all' in st.session_state:
        del st.session_state.predict_all

    # Performance summary
    if total_predictions > 0:
        st.markdown("---")
        st.subheader("📊 Performance Summary")

        accuracy = (correct_predictions / total_predictions) * 100

        perf_col1, perf_col2, perf_col3 = st.columns(3)

        with perf_col1:
            st.metric("Total Predictions", total_predictions)

        with perf_col2:
            st.metric("Correct Predictions", correct_predictions)

        with perf_col3:
            st.metric("Accuracy", f"{accuracy:.1f}%")

        # Visual feedback
        if accuracy >= 90:
            st.success(f"🎉 Excellent performance! The model achieved {accuracy:.1f}% accuracy on these samples.")
        elif accuracy >= 75:
            st.info(f"👍 Good performance! The model achieved {accuracy:.1f}% accuracy on these samples.")
        elif accuracy >= 50:
            st.warning(f"⚠️ Moderate performance. The model achieved {accuracy:.1f}% accuracy on these samples.")
        else:
            st.error(f"❌ Low performance. The model achieved only {accuracy:.1f}% accuracy on these samples.")

except Exception as e:
    st.error(f"Error loading samples: {str(e)}")

st.markdown("---")

# Information section
with st.expander("ℹ️ About the Sample Gallery"):
    st.markdown("""
    ### How It Works

    1. **Select Category**: Choose to view Fresh, Stale, or Both types of food images
    2. **Adjust Samples**: Use the slider to control how many images to display
    3. **Refresh**: Click refresh to load a new random set of images
    4. **Predict**: Click individual Predict buttons or use Predict All for batch processing

    ### Understanding the Results

    - **Green Box (🍎 Fresh)**: Model predicts the food is fresh
    - **Red Box (🤢 Stale)**: Model predicts the food is stale
    - **✓ Correct**: Model prediction matches the true label
    - **✗ Incorrect**: Model prediction differs from the true label

    ### Performance Metrics

    The accuracy shown is calculated only on the currently displayed samples.
    For overall model performance, check the Training History page.

    ### Tips

    - Try different sample sizes to see how the model performs across various images
    - Compare fresh vs stale predictions to understand model behavior
    - Use 'both' category to see how well the model distinguishes between classes
    """)

st.markdown("---")
st.caption("Food Quality Detection System | Sample Gallery")
