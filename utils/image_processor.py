"""
Image preprocessing utilities for Food Quality Detection App.

This module handles image preprocessing to match the training pipeline exactly:
- Resize to 224x224
- Normalize to [0, 1] by dividing by 255
- Add batch dimension for model input
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image


def preprocess_image(image_input, target_size=(224, 224)):
    """
    Preprocess image for model prediction.

    This function matches the training preprocessing pipeline exactly:
    - Resize to target_size (224x224)
    - Convert to numpy array
    - Normalize by dividing by 255.0 (rescale to [0, 1])
    - Add batch dimension

    Args:
        image_input: PIL Image object or file path string
        target_size: Tuple of (height, width), default (224, 224)

    Returns:
        numpy.ndarray: Preprocessed image array with shape (1, 224, 224, 3)
                      ready for model.predict()
    """
    # Load image if file path is provided
    if isinstance(image_input, str):
        img = image.load_img(image_input, target_size=target_size)
    else:
        # image_input is PIL Image
        img = image_input.resize(target_size)

    # Convert PIL Image to numpy array
    img_array = image.img_to_array(img)

    # Normalize to [0, 1] - must match training: rescale=1./255
    img_array = img_array / 255.0

    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def validate_image(uploaded_file):
    """
    Validate uploaded image file format.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        bool: True if file extension is valid, False otherwise
    """
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    if uploaded_file is None:
        return False

    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    return file_ext in allowed_extensions


def load_image_from_path(image_path):
    """
    Load image from file path and return as PIL Image.

    Args:
        image_path: Path to image file

    Returns:
        PIL.Image: Loaded image

    Raises:
        FileNotFoundError: If image file doesn't exist
        Exception: If image cannot be loaded
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        img = Image.open(image_path)
        # Convert to RGB if necessary (handle RGBA, grayscale, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    except Exception as e:
        raise Exception(f"Error loading image: {str(e)}")


def get_image_info(image_input):
    """
    Get information about an image.

    Args:
        image_input: PIL Image object or file path string

    Returns:
        dict: Image information (size, mode, format)
    """
    if isinstance(image_input, str):
        img = Image.open(image_input)
    else:
        img = image_input

    return {
        'width': img.size[0],
        'height': img.size[1],
        'mode': img.mode,
        'format': img.format if hasattr(img, 'format') else 'Unknown'
    }
