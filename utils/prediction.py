"""
Prediction utilities for Food Quality Detection App.

This module handles making predictions with the trained model,
matching the logic from train.py lines 175-177.
"""


def predict_food_quality(model, preprocessed_image):
    """
    Make prediction on preprocessed image.

    This function applies the same prediction logic as train.py:
    - Get sigmoid output value (0 to 1)
    - If < 0.5: Fresh (class 0)
    - If >= 0.5: Stale (class 1)
    - Calculate confidence accordingly

    Args:
        model: Loaded Keras model
        preprocessed_image: Preprocessed image array with shape (1, 224, 224, 3)

    Returns:
        dict: Prediction result with keys:
            - label: "Fresh" or "Stale"
            - confidence: Confidence percentage (0-100)
            - raw_prediction: Raw sigmoid output value (0-1)
    """
    # Get prediction from model
    # Output is sigmoid value between 0 and 1
    prediction_value = model.predict(preprocessed_image, verbose=0)[0][0]

    # Interpret result (matching train.py logic at lines 175-177)
    if prediction_value < 0.5:
        label = "Fresh"
        confidence = (1 - prediction_value) * 100
    else:
        label = "Stale"
        confidence = prediction_value * 100

    return {
        'label': label,
        'confidence': confidence,
        'raw_prediction': float(prediction_value)
    }


def get_prediction_color(label):
    """
    Get color code for displaying prediction result.

    Args:
        label: Prediction label ("Fresh" or "Stale")

    Returns:
        str: Color code for UI display
    """
    color_map = {
        "Fresh": "#28a745",  # Green
        "Stale": "#dc3545"   # Red
    }
    return color_map.get(label, "#6c757d")  # Default gray


def get_prediction_emoji(label):
    """
    Get emoji for prediction label.

    Args:
        label: Prediction label ("Fresh" or "Stale")

    Returns:
        str: Emoji character
    """
    emoji_map = {
        "Fresh": "🍎",
        "Stale": "🤢"
    }
    return emoji_map.get(label, "❓")


def interpret_confidence(confidence):
    """
    Interpret confidence level.

    Args:
        confidence: Confidence value (0-100)

    Returns:
        str: Interpretation of confidence level
    """
    if confidence >= 90:
        return "Very High"
    elif confidence >= 75:
        return "High"
    elif confidence >= 60:
        return "Moderate"
    elif confidence >= 50:
        return "Low"
    else:
        return "Very Low"
