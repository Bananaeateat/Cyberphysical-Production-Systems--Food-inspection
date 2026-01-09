"""
Quick test to verify the app components work
"""

import os
from utils.model_loader import load_model, check_model_exists
from utils.image_processor import preprocess_image, load_image_from_path
from utils.prediction import predict_food_quality
from utils.data_utils import get_dataset_stats, get_sample_images

print("=" * 50)
print("Testing Streamlit App Components")
print("=" * 50)

# Test 1: Check model exists
print("\n1. Checking model file...")
if check_model_exists():
    print("   OK: Model file exists")
else:
    print("   ERROR: Model file not found")

# Test 2: Load model
print("\n2. Loading model...")
model, error = load_model()
if error:
    print(f"   ERROR: {error}")
else:
    print("   OK: Model loaded successfully")

# Test 3: Get dataset stats
print("\n3. Getting dataset statistics...")
try:
    stats = get_dataset_stats()
    print(f"   OK: Total images: {stats['total']}")
    print(f"       Train: {stats['total_train']}, Test: {stats['total_test']}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 4: Get sample images
print("\n4. Getting sample images...")
try:
    samples = get_sample_images('fresh', num_samples=2)
    print(f"   OK: Found {len(samples)} sample images")
    if samples:
        print(f"       First sample: {os.path.basename(samples[0][0])}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 5: Load and preprocess an image
print("\n5. Testing image preprocessing...")
try:
    samples = get_sample_images('fresh', num_samples=1)
    if samples:
        img_path, true_label = samples[0]
        image = load_image_from_path(img_path)
        print(f"   OK: Loaded image: {os.path.basename(img_path)}")

        preprocessed = preprocess_image(image)
        print(f"   OK: Preprocessed shape: {preprocessed.shape}")
    else:
        print("   ERROR: No sample images found")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 6: Make prediction
print("\n6. Testing prediction...")
try:
    if model and samples:
        result = predict_food_quality(model, preprocessed)
        print(f"   OK: Prediction: {result['label']}")
        print(f"       Confidence: {result['confidence']:.2f}%")
        print(f"       Raw value: {result['raw_prediction']:.6f}")
    else:
        print("   SKIP: Model or samples not available")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "=" * 50)
print("Test Complete!")
print("=" * 50)
