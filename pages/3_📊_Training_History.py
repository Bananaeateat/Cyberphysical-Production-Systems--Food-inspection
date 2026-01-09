"""
Training History Page - View model training performance
"""

import streamlit as st
import os

# Page configuration
st.set_page_config(
    page_title="Training History",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Training History")
st.markdown("View the model's training performance metrics and curves")
st.markdown("---")

# Check if training history exists
history_path = "training_history.png"

if os.path.exists(history_path):
    st.success("✅ Training history available")

    # Display training curves
    st.subheader("📈 Training & Validation Curves")
    st.image(history_path, use_container_width=True)

    st.markdown("---")

    # Training information
    st.subheader("⚙️ Training Parameters")

    param_col1, param_col2, param_col3 = st.columns(3)

    with param_col1:
        st.metric("Image Size", "224 x 224")
        st.caption("Input image dimensions")

    with param_col2:
        st.metric("Batch Size", "32")
        st.caption("Number of images per batch")

    with param_col3:
        st.metric("Epochs", "10")
        st.caption("Training iterations")

    st.markdown("---")

    # Model architecture
    st.subheader("🏗️ Model Architecture")

    with st.expander("View Architecture Details", expanded=False):
        st.markdown("""
        ### Base Model: MobileNetV2
        - **Pretrained on**: ImageNet (1000 classes)
        - **Transfer Learning**: Base model frozen during training
        - **Input Shape**: (224, 224, 3)

        ### Custom Layers
        The model adds the following layers on top of MobileNetV2:

        1. **GlobalAveragePooling2D**
           - Reduces spatial dimensions to a single vector

        2. **BatchNormalization**
           - Normalizes activations for stable training

        3. **Dense Layer (256 units)**
           - Activation: ReLU
           - Followed by Dropout (0.5)

        4. **Dense Layer (128 units)**
           - Activation: ReLU
           - Followed by Dropout (0.3)

        5. **Output Layer (1 unit)**
           - Activation: Sigmoid
           - Binary classification output (0-1)

        ### Training Configuration
        - **Optimizer**: Adam (learning_rate=0.001)
        - **Loss Function**: Binary Crossentropy
        - **Metrics**: Accuracy
        """)

    st.markdown("---")

    # Data augmentation
    st.subheader("🔄 Data Augmentation")

    with st.expander("View Augmentation Details", expanded=False):
        st.markdown("""
        ### Training Data Augmentation
        The following augmentations were applied to training images:

        - **Rotation**: ±30 degrees
        - **Width Shift**: ±20%
        - **Height Shift**: ±20%
        - **Horizontal Flip**: Yes
        - **Zoom Range**: ±20%
        - **Brightness Range**: 0.8 - 1.2
        - **Fill Mode**: Nearest

        ### Normalization
        - **Rescale**: 1/255 (normalize to [0, 1])

        ### Validation Data
        - Only rescaling applied (no augmentation)
        - Ensures fair evaluation on original data distribution
        """)

    st.markdown("---")

    # Understanding the curves
    st.subheader("📖 Understanding the Curves")

    curve_col1, curve_col2 = st.columns(2)

    with curve_col1:
        st.markdown("""
        ### Accuracy Curves (Left)
        - **Blue Line**: Training accuracy
        - **Orange Line**: Validation accuracy

        **What to Look For:**
        - Both curves should increase over time
        - Curves should be close together
        - Gap indicates overfitting
        """)

    with curve_col2:
        st.markdown("""
        ### Loss Curves (Right)
        - **Blue Line**: Training loss
        - **Orange Line**: Validation loss

        **What to Look For:**
        - Both curves should decrease over time
        - Lower values indicate better performance
        - Large gap indicates overfitting
        """)

    # Performance indicators
    st.info("""
    **Good Model Indicators:**
    - ✅ Training and validation accuracy both high (>85%)
    - ✅ Small gap between training and validation metrics
    - ✅ Smooth curves without erratic jumps
    - ✅ Validation loss not increasing while training loss decreases
    """)

else:
    st.warning("⚠️ Training history not found")

    st.info("""
    ### The training history chart is not available yet.

    The chart `training_history.png` will be generated automatically when you train the model.

    ### To Generate Training History:

    1. Open a terminal in the project directory
    2. Run the training script:
       ```bash
       python train.py
       ```
    3. Wait for training to complete (10 epochs)
    4. The chart will be saved as `training_history.png`
    5. Refresh this page to view the chart

    ### What You'll See:

    The training history chart shows:
    - **Accuracy over epochs**: How well the model classifies images
    - **Loss over epochs**: How much error the model makes
    - **Training vs Validation**: Comparison of performance on training and test data

    These curves help understand if the model is learning properly or if there are issues
    like overfitting (good on training, poor on validation).
    """)

    # Show example of what to expect
    st.markdown("---")
    st.subheader("📚 What to Expect")

    expect_col1, expect_col2 = st.columns(2)

    with expect_col1:
        st.markdown("""
        **Expected Accuracy:**
        - Training: 85-95%
        - Validation: 80-90%
        - Gap: <5%
        """)

    with expect_col2:
        st.markdown("""
        **Expected Loss:**
        - Training: 0.1-0.3
        - Validation: 0.2-0.4
        - Decreasing trend
        """)

st.markdown("---")

# Additional resources
with st.expander("📚 Learn More About Training Metrics"):
    st.markdown("""
    ### Key Concepts

    **Accuracy**
    - Percentage of correctly classified images
    - Range: 0-100% (higher is better)
    - Formula: (Correct Predictions / Total Predictions) × 100

    **Loss (Binary Crossentropy)**
    - Measures how wrong the predictions are
    - Range: 0 to infinity (lower is better)
    - Quantifies the difference between predicted and true labels

    **Training vs Validation**
    - **Training Set**: Used to update model weights
    - **Validation Set**: Used to evaluate performance
    - Gap between them indicates generalization ability

    **Epochs**
    - One complete pass through the training dataset
    - More epochs = more learning opportunities
    - Too many epochs can lead to overfitting

    **Overfitting**
    - Model performs well on training data but poorly on new data
    - Indicated by: high training accuracy, low validation accuracy
    - Solutions: more data, augmentation, dropout, regularization

    **Underfitting**
    - Model performs poorly on both training and validation data
    - Indicated by: low accuracy on both sets
    - Solutions: more complex model, more training epochs, less regularization
    """)

st.markdown("---")
st.caption("Food Quality Detection System | Training History")
