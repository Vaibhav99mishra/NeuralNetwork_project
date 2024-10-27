
# Enhancing Plant Stress Detection through Denoising of Audio Signals and Advanced Classification Models

This project aims to classify plant stress conditions by converting audio recordings into spectrogram images and applying deep learning techniques for image classification. By leveraging EfficientNet and denoising techniques, this approach enhances the accuracy of identifying various plant stress signals from audio spectrograms. This repository provides a detailed overview of the workflow, including audio-to-image conversion, denoising, model training, and evaluation.

## Team Members
- Varun Biyyala
- Vaibhav Mishra

## Overview

### 1. Audio-to-Image Conversion
The audio files in different plant stress conditions are converted into spectrogram images using the following steps:
- **Spectrogram Generation**: Each audio file is transformed into a spectrogram using `torchaudio` and saved as an image.
- **Visualization**: Sample spectrogram images for stress conditions like "Tobacco Cut" and "Tomato Dry" are displayed for reference.

### 2. Image Denoising
To improve the quality of spectrogram images, denoising techniques are applied:
- **Denoising Function**: `cv2.fastNlMeansDenoising` is used to reduce noise in the spectrogram images.
- **Custom Transform**: A `DenoiseImage` transformation is created to integrate denoising as a preprocessing step in the data pipeline.

### 3. Model Architecture
We used **EfficientNet** (variants B3, B4, and B5) for classification. EfficientNet's pre-trained weights are fine-tuned to classify images into six classes representing different plant stress conditions.

### 4. Model Training and Evaluation
- **Data Augmentation**: Applied resizing, normalization, and denoising transformations.
- **Training Setup**: The model was trained for multiple epochs using cross-entropy loss and Adam optimizer. A learning rate scheduler was used to improve convergence.
- **Evaluation**: Accuracy on test sets is tracked after each epoch, achieving up to **98.77% accuracy** in distinguishing between "Tomato Cut" and "Greenhouse Noises."

### 5. Performance Test
We tested the model's performance on specific stress conditions, such as "Tomato Cut" vs. "Greenhouse Noises," demonstrating the model's robustness in distinguishing similar audio conditions after denoising.

## Key Results

- **Training Accuracy**: Achieved high accuracy over multiple epochs with EfficientNet B4, peaking at **76.19%** accuracy for the full six-class classification and **98.77%** for specific stress conditions.
- **Denoising Impact**: The denoising step significantly improved the clarity of spectrogram images, contributing to better model accuracy.

## Repository Structure

- **Data Preparation**: Audio-to-image conversion and denoising.
- **Model Training**: Training EfficientNet on denoised spectrogram images.
- **Evaluation**: Evaluation script to measure accuracy and other metrics.

## Dataset

- [Plant Sounds Dataset](https://drive.google.com/drive/folders/101xgZQ50q9Fsv3T66WZfyXj2jS1D0Do1?usp=drive_link): Contains audio files categorized by different plant stress conditions.

## Pre-trained Model

- Download the pre-trained model [here](https://drive.google.com/file/d/1r9E_4b_1_SwnUY5K-LL8Vjj9PIk0VM12/view?usp=sharing).

## Research Paper

For a deeper understanding of the research, refer to our paper on [ResearchGate](https://www.researchgate.net/publication/376685059_Enhancing_Plant_Stress_Detection_through_Denoising_of_Audio_Signals_and_Advanced_Classification_Models).

## How to Use

1. **Data Preparation**: Convert audio files to spectrogram images and denoise them.
2. **Model Training**: Train the model using the provided dataset and scripts.
3. **Evaluation**: Evaluate the model on test data to check the classification accuracy.

## Requirements

- Python 3.x
- PyTorch
- torchaudio
- torchvision
- OpenCV
- Matplotlib
- Scikit-learn

Install dependencies:
```bash
pip install torch torchaudio torchvision opencv-python-headless matplotlib scikit-learn
```

## Example Usage

```python
# Convert audio files to spectrogram images
audio_folder_path = '/path/to/audio/files'
image_folder_path = '/path/to/save/images'
audio_to_image(audio_folder_path, image_folder_path)

# Train EfficientNet on the generated images
model_name = 'efficientnet_b4'
num_classes = 6
model = initialize_efficientnet(model_name, num_classes)
# Continue with training and evaluation...
```

## Results Visualization

Sample spectrogram images for various stress conditions:
- **Tobacco Cut**
- **Tobacco Dry**
- **Tomato Cut**
- **Tomato Dry**

Each spectrogram represents different audio signatures, aiding the model in differentiating between stress conditions.

---

This README provides a summary of the project's steps and results. For more detailed instructions, please refer to the code and additional comments in the provided Jupyter notebooks and scripts.
