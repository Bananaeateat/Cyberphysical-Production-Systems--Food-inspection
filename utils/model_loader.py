"""
Model loading utilities with caching for Food Quality Detection App.
"""

import os
import streamlit as st
import tensorflow as tf


@st.cache_resource
def load_model():
    """
    Load the trained model with caching.

    This function uses Streamlit's cache_resource decorator to load the model
    once per session and persist it across page navigation.

    Returns:
        tuple: (model, error_message)
            - model: Loaded Keras model if successful, None otherwise
            - error_message: Error description if failed, None otherwise
    """
    model_path = "models/food_quality_detector.h5"

    # Check if model file exists
    if not os.path.exists(model_path):
        return None, "Model file not found. Please train the model first by running: python train.py"

    # Try to load the model
    try:
        model = tf.keras.models.load_model(model_path)
        return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


def check_model_exists():
    """
    Quick check if model file exists without loading it.

    Returns:
        bool: True if model file exists, False otherwise
    """
    return os.path.exists("models/food_quality_detector.h5")


def get_model_info():
    """
    Get information about the model architecture.

    Returns:
        dict: Model information including architecture details
    """
    return {
        'base_model': 'MobileNetV2',
        'pretrained_on': 'ImageNet',
        'image_size': '224x224',
        'input_channels': 3,
        'output_type': 'Binary Classification',
        'classes': ['Fresh (0)', 'Stale (1)'],
        'activation': 'Sigmoid'
    }
