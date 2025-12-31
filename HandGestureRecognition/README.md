# Hand Gesture Recognition

A deep learning project that uses Convolutional Neural Networks (CNN) to recognize and classify hand gestures from images.

## Overview

This project implements a CNN-based image classification system to identify different hand gestures. The model processes grayscale images of hand gestures and predicts the corresponding gesture class using a multi-layer convolutional architecture.

## Dataset

- **Source**: Leap Gesture Recognition dataset
- **Format**: PNG images of hand gestures
- **Processing**: Images are converted to grayscale and resized to 320x120 pixels
- **Categories**: Multiple gesture classes (automatically extracted from directory structure)

## Model Architecture

### Convolutional Neural Network
The model consists of the following layers:

1. **Conv2D Layer 1**: 32 filters, 5x5 kernel, ReLU activation
2. **MaxPooling2D**: 2x2 pool size
3. **Conv2D Layer 2**: 64 filters, 3x3 kernel, ReLU activation
4. **MaxPooling2D**: 2x2 pool size
5. **Conv2D Layer 3**: 64 filters, 3x3 kernel, ReLU activation
6. **MaxPooling2D**: 2x2 pool size
7. **Flatten Layer**: Converts 2D features to 1D
8. **Dense Layer**: 128 neurons, ReLU activation
9. **Output Layer**: Softmax activation for multi-class classification

### Model Configuration
- **Input Shape**: (120, 320, 1) - Grayscale images
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy

## Training Details

- **Train/Test Split**: 70/30 (random_state=42)
- **Epochs**: 5
- **Batch Size**: 64
- **Validation**: Performed on test set during training

## Features

1. **Image Preprocessing**:
   - Automatic image loading from directory structure
   - Grayscale conversion
   - Resizing to uniform dimensions (320x120)
   - Label extraction from directory names

2. **Model Training**:
   - Efficient batch processing
   - Validation during training
   - Model saving for future use

3. **Evaluation & Visualization**:
   - Accuracy and loss plots over epochs
   - Confusion matrix for detailed performance analysis
   - Sample prediction visualization (9 test images)
   - Color-coded predictions (blue = correct, red = incorrect)

## Technologies Used

- **TensorFlow/Keras** - Deep learning framework
- **OpenCV (cv2)** - Image processing
- **NumPy** - Numerical operations
- **Pandas** - Data manipulation and confusion matrix display
- **Matplotlib** - Visualization and plotting
- **scikit-learn** - Train/test split and metrics

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install tensorflow opencv-python numpy pandas matplotlib scikit-learn
   ```

2. **Prepare Dataset**:
   - Organize gesture images in directories named with format: `<label>_<description>`
   - Update `data_path` variable in the script to point to your dataset location

3. **Run the Script**:
   ```bash
   python final_code.py
   ```

4. **Output**:
   - Model will train for 5 epochs
   - Training and validation accuracy/loss will be displayed
   - Saved model: `handrecognition_model.h5`
   - Visualization plots will be generated
   - Confusion matrix will be printed
   - Sample predictions on test images will be shown

## Model Output

The trained model produces:
- **Test Accuracy**: Displayed as percentage
- **Training History**: Accuracy and loss curves for both training and validation
- **Confusion Matrix**: Detailed breakdown of predictions vs actual labels
- **Visual Validation**: Grid display of 9 sample predictions with confidence scores

## Results & Visualization

1. **Accuracy Plot**: Shows training vs validation accuracy across epochs
2. **Loss Plot**: Shows training vs validation loss across epochs
3. **Confusion Matrix**: Tabular view of classification performance per class
4. **Prediction Samples**: Visual grid showing 9 test images with:
   - Predicted class and confidence percentage
   - True class label
   - Color coding for correct/incorrect predictions

## Future Improvements

- Data augmentation for better generalization
- Increased training epochs with early stopping
- Transfer learning using pre-trained models
- Real-time gesture recognition from webcam
- Mobile deployment for gesture-based interfaces

## Use Cases

- Human-Computer Interaction (HCI)
- Sign language recognition
- Touchless control systems
- Gaming interfaces
- Assistive technology

## License

Educational project for machine learning portfolio.
