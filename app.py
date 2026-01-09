"""
Food Quality Detection System - Main App
A Streamlit application for detecting fresh vs stale food using deep learning.
"""

import streamlit as st
from utils.model_loader import check_model_exists, get_model_info
from utils.data_utils import get_dataset_stats

# Page configuration
st.set_page_config(
    page_title="Food Quality Detection",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("🍎 Food Quality Detection System")
st.markdown("---")

# Welcome section
st.markdown("""
### Welcome to the Food Quality Detection System

This interactive application uses deep learning to classify food images as **Fresh** or **Stale**.
The model is based on **MobileNetV2** architecture with transfer learning from ImageNet.

#### How to Use:
1. Navigate to **🔍 Prediction** to upload and classify your own images
2. Check out the **🖼️ Sample Gallery** to see the model in action on test data
3. View **📊 Training History** to see how the model was trained
4. Explore **📈 Dataset Statistics** to understand the data distribution
""")

st.markdown("---")

# Model status check
st.subheader("📊 System Status")

col1, col2, col3 = st.columns(3)

# Check model status
model_exists = check_model_exists()

with col1:
    if model_exists:
        st.success("✅ Model Loaded")
        st.caption("Ready for predictions")
    else:
        st.error("❌ Model Not Found")
        st.caption("Please train the model first")

# Get dataset statistics
try:
    stats = get_dataset_stats()
    with col2:
        st.info(f"📁 Dataset: {stats['total']:,} images")
        st.caption(f"Train: {stats['total_train']:,} | Test: {stats['total_test']:,}")

    with col3:
        st.info("🎯 Binary Classification")
        st.caption("Fresh vs Stale")
except Exception as e:
    with col2:
        st.warning("⚠️ Dataset Not Found")
        st.caption("Please organize data first")
    with col3:
        st.info("🎯 Binary Classification")
        st.caption("Fresh vs Stale")

st.markdown("---")

# Quick stats
if model_exists:
    st.subheader("🔧 Model Information")

    model_info = get_model_info()

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.metric("Base Model", model_info['base_model'])
        st.metric("Pretrained On", model_info['pretrained_on'])

    with info_col2:
        st.metric("Input Size", model_info['image_size'])
        st.metric("Input Channels", model_info['input_channels'])

    with info_col3:
        st.metric("Output Type", model_info['output_type'])
        st.metric("Activation", model_info['activation'])

    st.markdown("---")

# Instructions
st.subheader("🚀 Getting Started")

if not model_exists:
    st.warning("""
    **Model not found!** Please train the model before using the app.

    To train the model:
    1. Open a terminal in the project directory
    2. Run: `python train.py`
    3. Wait for training to complete (approximately 10 epochs)
    4. Refresh this page

    The model will be saved to `models/food_quality_detector.h5`
    """)
else:
    st.success("""
    **System ready!** Navigate to the pages using the sidebar:

    - **🔍 Prediction**: Upload your own food images for classification
    - **🖼️ Sample Gallery**: Test the model on sample images from the dataset
    - **📊 Training History**: View training performance metrics and curves
    - **📈 Dataset Statistics**: Explore the dataset composition and distribution
    """)

st.markdown("---")

# Dataset overview
st.subheader("📦 Dataset Overview")

try:
    stats = get_dataset_stats()

    if stats['total'] > 0:
        # Display stats
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            st.metric("Total Images", f"{stats['total']:,}")

        with metric_col2:
            st.metric("Training Set", f"{stats['total_train']:,}",
                     f"{stats['train_percentage']:.1f}%")

        with metric_col3:
            st.metric("Test Set", f"{stats['total_test']:,}",
                     f"{stats['test_percentage']:.1f}%")

        with metric_col4:
            st.metric("Classes", "2", "Fresh & Stale")

        # Class distribution
        st.markdown("#### Class Distribution")
        dist_col1, dist_col2 = st.columns(2)

        with dist_col1:
            st.write("**Training Set**")
            st.write(f"- Fresh: {stats['train_fresh']:,} ({stats['train_fresh_percentage']:.1f}%)")
            st.write(f"- Stale: {stats['train_stale']:,} ({stats['train_stale_percentage']:.1f}%)")

        with dist_col2:
            st.write("**Test Set**")
            st.write(f"- Fresh: {stats['test_fresh']:,} ({stats['test_fresh_percentage']:.1f}%)")
            st.write(f"- Stale: {stats['test_stale']:,} ({stats['test_stale_percentage']:.1f}%)")
    else:
        st.warning("""
        **Dataset not found!** Please organize the dataset first.

        To organize the dataset:
        1. Ensure raw data is in the `dataset/` directory
        2. Run: `python organize_data.py`
        3. Refresh this page
        """)
except Exception as e:
    st.error(f"Error loading dataset statistics: {str(e)}")

st.markdown("---")

# Footer
st.markdown("""
### 🛠️ Technology Stack

- **Framework**: TensorFlow / Keras
- **Base Model**: MobileNetV2 (Transfer Learning)
- **Frontend**: Streamlit
- **Dataset**: 30,000+ food images (fresh & stale)

#### About This Project
This system demonstrates the application of deep learning for food quality assessment.
The model uses transfer learning with MobileNetV2 to achieve high accuracy in
distinguishing between fresh and stale food items.

---
*Built with Streamlit and TensorFlow*
""")
