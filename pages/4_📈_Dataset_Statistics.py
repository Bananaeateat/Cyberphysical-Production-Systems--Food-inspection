"""
Dataset Statistics Page - Explore dataset composition and distribution
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import get_dataset_stats, check_data_directories

# Page configuration
st.set_page_config(
    page_title="Dataset Statistics",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Dataset Statistics")
st.markdown("Explore the composition and distribution of the food quality dataset")
st.markdown("---")

# Check data directories
dir_status = check_data_directories()
all_exist = all(dir_status.values())

if not all_exist:
    st.warning("⚠️ Some data directories are missing")

    missing_dirs = [key for key, exists in dir_status.items() if not exists]
    st.error(f"Missing directories: {', '.join(missing_dirs)}")

    st.info("""
    ### Setup Required

    Please organize the dataset first:

    1. Ensure raw data is in the `dataset/` directory
    2. Run: `python organize_data.py`
    3. This will create the proper directory structure:
       - `data/train/fresh/`
       - `data/train/stale/`
       - `data/test/fresh/`
       - `data/test/stale/`
    4. Refresh this page
    """)
    st.stop()

# Get statistics
try:
    stats = get_dataset_stats()

    if stats['total'] == 0:
        st.error("❌ No images found in dataset")
        st.info("Please run `python organize_data.py` to prepare the dataset")
        st.stop()

    st.success(f"✅ Dataset loaded: {stats['total']:,} images")
    st.markdown("---")

    # Overall statistics
    st.subheader("📊 Overview")

    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)

    with overview_col1:
        st.metric(
            "Total Images",
            f"{stats['total']:,}",
            help="Total number of images in the dataset"
        )

    with overview_col2:
        st.metric(
            "Training Images",
            f"{stats['total_train']:,}",
            f"{stats['train_percentage']:.1f}%",
            help="Images used for training the model"
        )

    with overview_col3:
        st.metric(
            "Test Images",
            f"{stats['total_test']:,}",
            f"{stats['test_percentage']:.1f}%",
            help="Images used for evaluating the model"
        )

    with overview_col4:
        st.metric(
            "Classes",
            "2",
            "Fresh & Stale",
            help="Binary classification problem"
        )

    st.markdown("---")

    # Class distribution
    st.subheader("🎯 Class Distribution")

    class_col1, class_col2 = st.columns(2)

    with class_col1:
        st.markdown("### Training Set")

        # Training set metrics
        train_metrics_col1, train_metrics_col2 = st.columns(2)

        with train_metrics_col1:
            st.metric(
                "Fresh Images",
                f"{stats['train_fresh']:,}",
                f"{stats['train_fresh_percentage']:.1f}%"
            )

        with train_metrics_col2:
            st.metric(
                "Stale Images",
                f"{stats['train_stale']:,}",
                f"{stats['train_stale_percentage']:.1f}%"
            )

        # Training pie chart
        train_fig = go.Figure(data=[go.Pie(
            labels=['Fresh', 'Stale'],
            values=[stats['train_fresh'], stats['train_stale']],
            marker=dict(colors=['#28a745', '#dc3545']),
            hole=0.4
        )])
        train_fig.update_layout(
            title="Training Set Distribution",
            height=300
        )
        st.plotly_chart(train_fig, use_container_width=True)

    with class_col2:
        st.markdown("### Test Set")

        # Test set metrics
        test_metrics_col1, test_metrics_col2 = st.columns(2)

        with test_metrics_col1:
            st.metric(
                "Fresh Images",
                f"{stats['test_fresh']:,}",
                f"{stats['test_fresh_percentage']:.1f}%"
            )

        with test_metrics_col2:
            st.metric(
                "Stale Images",
                f"{stats['test_stale']:,}",
                f"{stats['test_stale_percentage']:.1f}%"
            )

        # Test pie chart
        test_fig = go.Figure(data=[go.Pie(
            labels=['Fresh', 'Stale'],
            values=[stats['test_fresh'], stats['test_stale']],
            marker=dict(colors=['#28a745', '#dc3545']),
            hole=0.4
        )])
        test_fig.update_layout(
            title="Test Set Distribution",
            height=300
        )
        st.plotly_chart(test_fig, use_container_width=True)

    st.markdown("---")

    # Train/Test split visualization
    st.subheader("🔀 Train/Test Split")

    split_col1, split_col2 = st.columns([2, 1])

    with split_col1:
        # Bar chart showing train/test split
        split_df = pd.DataFrame({
            'Set': ['Training', 'Test'],
            'Fresh': [stats['train_fresh'], stats['test_fresh']],
            'Stale': [stats['train_stale'], stats['test_stale']]
        })

        split_fig = go.Figure()
        split_fig.add_trace(go.Bar(
            name='Fresh',
            x=split_df['Set'],
            y=split_df['Fresh'],
            marker_color='#28a745'
        ))
        split_fig.add_trace(go.Bar(
            name='Stale',
            x=split_df['Set'],
            y=split_df['Stale'],
            marker_color='#dc3545'
        ))

        split_fig.update_layout(
            title='Images per Set and Class',
            xaxis_title='Dataset Split',
            yaxis_title='Number of Images',
            barmode='group',
            height=400
        )
        st.plotly_chart(split_fig, use_container_width=True)

    with split_col2:
        st.markdown("### Split Ratio")

        # Train/Test pie chart
        split_ratio_fig = go.Figure(data=[go.Pie(
            labels=['Training', 'Test'],
            values=[stats['total_train'], stats['total_test']],
            marker=dict(colors=['#007bff', '#17a2b8']),
            hole=0.4
        )])
        split_ratio_fig.update_layout(height=400)
        st.plotly_chart(split_ratio_fig, use_container_width=True)

        # Split ratio info
        train_ratio = stats['total_train'] / stats['total'] if stats['total'] > 0 else 0
        test_ratio = stats['total_test'] / stats['total'] if stats['total'] > 0 else 0

        st.info(f"""
        **Split Ratio**: {train_ratio:.0%} / {test_ratio:.0%}

        This is a standard split for machine learning,
        providing enough data for training while
        maintaining a representative test set.
        """)

    st.markdown("---")

    # Detailed statistics table
    st.subheader("📋 Detailed Statistics")

    # Create detailed dataframe
    detailed_df = pd.DataFrame({
        'Category': ['Training - Fresh', 'Training - Stale', 'Test - Fresh', 'Test - Stale'],
        'Count': [stats['train_fresh'], stats['train_stale'], stats['test_fresh'], stats['test_stale']],
        'Percentage': [
            f"{stats['train_fresh_percentage']:.2f}%",
            f"{stats['train_stale_percentage']:.2f}%",
            f"{stats['test_fresh_percentage']:.2f}%",
            f"{stats['test_stale_percentage']:.2f}%"
        ]
    })

    st.dataframe(
        detailed_df,
        use_container_width=True,
        hide_index=True
    )

    # Class balance analysis
    st.markdown("---")
    st.subheader("⚖️ Class Balance Analysis")

    balance_col1, balance_col2 = st.columns(2)

    with balance_col1:
        # Calculate imbalance ratio for training set
        train_imbalance = max(stats['train_fresh'], stats['train_stale']) / min(stats['train_fresh'], stats['train_stale'])

        st.markdown("### Training Set Balance")
        st.metric("Imbalance Ratio", f"{train_imbalance:.2f}:1")

        if train_imbalance < 1.2:
            st.success("✅ Well balanced - Excellent for training")
        elif train_imbalance < 1.5:
            st.info("ℹ️ Slightly imbalanced - Still good for training")
        elif train_imbalance < 2.0:
            st.warning("⚠️ Moderately imbalanced - Consider class weights")
        else:
            st.error("❌ Highly imbalanced - Class weights recommended")

    with balance_col2:
        # Calculate imbalance ratio for test set
        test_imbalance = max(stats['test_fresh'], stats['test_stale']) / min(stats['test_fresh'], stats['test_stale'])

        st.markdown("### Test Set Balance")
        st.metric("Imbalance Ratio", f"{test_imbalance:.2f}:1")

        if test_imbalance < 1.2:
            st.success("✅ Well balanced - Reliable evaluation")
        elif test_imbalance < 1.5:
            st.info("ℹ️ Slightly imbalanced - Evaluation is still reliable")
        elif test_imbalance < 2.0:
            st.warning("⚠️ Moderately imbalanced - Consider this in evaluation")
        else:
            st.error("❌ Highly imbalanced - Evaluation may be skewed")

    st.markdown("---")

    # Dataset quality indicators
    st.subheader("✨ Dataset Quality Indicators")

    quality_col1, quality_col2, quality_col3 = st.columns(3)

    with quality_col1:
        size_score = "✅ Excellent" if stats['total'] >= 20000 else "⚠️ Moderate" if stats['total'] >= 10000 else "❌ Small"
        st.metric("Dataset Size", size_score)
        st.caption(f"{stats['total']:,} total images")

    with quality_col2:
        split_score = "✅ Optimal" if 0.15 <= test_ratio <= 0.25 else "ℹ️ Acceptable"
        st.metric("Split Ratio", split_score)
        st.caption(f"{test_ratio:.1%} test set")

    with quality_col3:
        avg_imbalance = (train_imbalance + test_imbalance) / 2
        balance_score = "✅ Balanced" if avg_imbalance < 1.3 else "⚠️ Imbalanced"
        st.metric("Class Balance", balance_score)
        st.caption(f"{avg_imbalance:.2f}:1 ratio")

except Exception as e:
    st.error(f"Error loading statistics: {str(e)}")

st.markdown("---")

# Information section
with st.expander("ℹ️ Understanding Dataset Statistics"):
    st.markdown("""
    ### Why Dataset Statistics Matter

    **Dataset Size**
    - Larger datasets generally lead to better model performance
    - More data helps the model learn diverse patterns
    - 20,000+ images is considered good for this type of task

    **Train/Test Split**
    - Training set: Used to update model weights
    - Test set: Used to evaluate model performance
    - Common splits: 80/20, 75/25, or 70/30
    - Test set should never be used for training

    **Class Balance**
    - Balanced classes ensure fair learning
    - Imbalance ratio < 1.5:1 is generally acceptable
    - Severe imbalance (>2:1) may require:
      - Class weights during training
      - Data augmentation for minority class
      - Different evaluation metrics (F1-score, etc.)

    ### This Dataset

    This dataset contains images of various food items labeled as:
    - **Fresh**: Food that is safe to eat, with good appearance
    - **Stale**: Food showing signs of spoilage or decay

    The images come from multiple food categories including fruits,
    vegetables, and other food items, providing diversity for the model
    to learn robust features for freshness detection.
    """)

st.markdown("---")
st.caption("Food Quality Detection System | Dataset Statistics")
