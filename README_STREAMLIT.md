# Food Quality Detection Streamlit App

An interactive web application for detecting fresh vs stale food using deep learning.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (if not already done)

```bash
python train.py
```

This will:
- Train a MobileNetV2-based model
- Save the model to `models/food_quality_detector.h5`
- Generate training curves in `training_history.png`

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## App Features

### 🏠 Homepage (app.py)
- System status overview
- Model and dataset information
- Quick statistics dashboard
- Getting started guide

### 🔍 Prediction Page
- **Upload your own images** for classification
- Get instant fresh/stale predictions
- View confidence scores
- See detailed technical information

### 🖼️ Sample Gallery
- Test the model on sample images from the test dataset
- Select fresh, stale, or both categories
- Batch prediction on multiple images
- View accuracy on displayed samples

### 📊 Training History
- View training and validation curves
- Understand model performance over epochs
- Learn about model architecture
- See data augmentation details

### 📈 Dataset Statistics
- Explore dataset composition
- View class distribution (fresh vs stale)
- Analyze train/test split
- Check class balance
- Interactive charts and visualizations

## Project Structure

```
Food_quality_project/
├── app.py                          # Main homepage
├── requirements.txt                # Python dependencies
├── train.py                        # Model training script
├── organize_data.py                # Dataset organization script
├── utils/                          # Utility modules
│   ├── __init__.py
│   ├── model_loader.py            # Model loading with caching
│   ├── image_processor.py         # Image preprocessing
│   ├── prediction.py              # Prediction logic
│   └── data_utils.py              # Data statistics
├── pages/                          # Streamlit pages
│   ├── 1_🔍_Prediction.py         # Image upload & prediction
│   ├── 2_🖼️_Sample_Gallery.py    # Test set gallery
│   ├── 3_📊_Training_History.py   # Training charts
│   └── 4_📈_Dataset_Statistics.py # Dataset stats
├── models/
│   └── food_quality_detector.h5   # Trained model (after training)
├── data/
│   ├── train/
│   │   ├── fresh/                 # Fresh training images
│   │   └── stale/                 # Stale training images
│   └── test/
│       ├── fresh/                 # Fresh test images
│       └── stale/                 # Stale test images
└── training_history.png           # Training curves (after training)
```

## Model Information

- **Architecture**: MobileNetV2 (Transfer Learning)
- **Pretrained On**: ImageNet
- **Input Size**: 224 x 224 pixels
- **Classes**: 2 (Fresh, Stale)
- **Output**: Binary classification with sigmoid activation

## Usage Tips

### For Best Prediction Results:
1. Use clear, well-lit images
2. Ensure the food item is in focus
3. Avoid images with heavy filters
4. Close-up shots work best
5. Single food items recommended

### Understanding Predictions:
- **Fresh (🍎)**: Prediction value < 0.5 → Food appears safe to eat
- **Stale (🤢)**: Prediction value ≥ 0.5 → Food shows signs of spoilage
- **Confidence**: Higher percentage = more confident prediction

## Troubleshooting

### Model Not Found Error
```
Error: Model file not found
```
**Solution**: Run `python train.py` to train the model first.

### Dataset Not Found Error
```
Error: Dataset not found
```
**Solution**: Run `python organize_data.py` to organize the dataset.

### Import Errors
```
Error: No module named 'streamlit'
```
**Solution**: Run `pip install -r requirements.txt`

### Page Not Loading
- Ensure you're in the project directory
- Check that all files are in place
- Restart the Streamlit server: `Ctrl+C` then `streamlit run app.py`

## Technical Details

### Image Preprocessing
All images are preprocessed to match the training pipeline:
1. Resize to 224 x 224 pixels
2. Convert to numpy array
3. Normalize to [0, 1] by dividing by 255
4. Add batch dimension

### Model Loading
- Uses `@st.cache_resource` for efficient caching
- Model loaded once per session
- Persists across page navigation

### Prediction Logic
Matches the training script logic:
```python
if prediction < 0.5:
    label = "Fresh"
    confidence = (1 - prediction) * 100
else:
    label = "Stale"
    confidence = prediction * 100
```

## Dataset Information

- **Source**: [Fresh and Stale Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification/data)
- **Total Images**: ~30,000+
- **Training Set**: ~78% (23,619 images)
- **Test Set**: ~22% (6,738 images)
- **Classes**: Fresh and Stale
- **Food Categories**: Various (apples, banana, cucumber, tomato, etc.)

## Performance

- **Model Loading**: 5-10 seconds (first time)
- **Prediction**: < 1 second per image
- **Gallery Loading**: 2-3 seconds
- **Page Navigation**: Instant (cached model)

## Development

### Adding New Features
1. Create new utility functions in `utils/`
2. Add new pages in `pages/` directory
3. Follow naming convention: `N_emoji_Name.py`
4. Import utilities using relative imports

### Customization
- Colors can be adjusted in prediction.py (`get_prediction_color`)
- Thresholds can be modified in prediction logic
- UI layout can be customized in each page

## Credits

- **Framework**: Streamlit
- **Deep Learning**: TensorFlow / Keras
- **Model**: MobileNetV2 (Transfer Learning)
- **Visualization**: Plotly

## Notes

- This is a demo/educational tool
- Always use your own judgment for food safety
- AI predictions are not a substitute for proper food safety practices
- When in doubt, discard questionable food items

---

**Built with Streamlit and TensorFlow**
