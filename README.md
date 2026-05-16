# Crop-Query

A Multilingual Vision-Language System for Intelligent Agricultural Monitoring.

## Overview
Crop-Query is a specialized deep learning pipeline designed to classify an extensive range of crop types and crop diseases (65 distinct classes). The project utilizes a ResNet50-based architecture trained progressively in three distinct phases to ensure optimal feature extraction and high classification accuracy.

## Overview
Crop-Query is a multi-stage deep learning application tailored for identifying 65 distinct agricultural crop types and diseases. 

## Project Structure
- **`cropquery.ipynb`**: **[Primary Focus]** The main notebook of the project. It handles data preprocessing, model initialization (ResNet50), loading pre-trained weights, extracting predictions, and running evaluations. **(See Step-by-Step Usage Below)**
- **Helper Files (`app.py`, `src/`, `data/`)**: Scripts and mappings (like `class_names.json`) that support the background processing.
- **Historical Models & Training (`notebooks/training/`, `models/`)**: Previous iterations and phase-based Jupyter notebooks used solely to reach the final trained model weights (`phase3_final.pth`).

## Setup & Dependencies
```bash
pip install torch torchvision numpy pandas matplotlib jupyter
```

## Step-by-Step: `cropquery.ipynb` Workflow

The entire lifecycle of model evaluation and inference happens within **`cropquery.ipynb`**. Here is the step-by-step logic it executes:

1. **Environment Setup & Imports**: Loads `torch`, `torchvision`, and data science libraries necessary for plotting and array manipulation.
2. **Dataset & Class Loading**: Reads `class_names.json` to map the 65 agricultural classes (e.g., Apple Scab, Corn Healthy) to numeric keys.
3. **Image Preprocessing**: Defines the Torchvision `transforms`, standardizing spatial dimensions and normalizing pixel tensor values for the ResNet50 architecture.
4. **Model Initialization**: Instantiates the ResNet-50 backbone modified to output our precise 65 classification classes instead of the default ImageNet 1k.
5. **Loading Configurations**: Automatically loads the final optimal weights (`phase3_final.pth`) from disk into the model structure to skip the lengthy training phase.
6. **Inference & Visualization**: Processes the input queries (test validation images), evaluates them through the model securely without backpropagation (`torch.no_grad()`), and plots the results inline natively via Matplotlib directly beneath the cell highlighting predicted ailments or crop health.

## Supported Crops
The system detects health issues and states across a wide variety of flora, including but not limited to:
- Apples, Cherries, Peaches, Grapes, Strawberries
- Bell Peppers, Tomatoes, Potatoes, Pumpkins, Peas, Melons
- Corn, Rice, Wheat, Sugarcane
- Coffee, Cashew, Cassava
