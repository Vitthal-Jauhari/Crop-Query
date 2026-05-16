# Crop-Query

A Multilingual Vision-Language System for Intelligent Agricultural Monitoring.

## Overview
Crop-Query is a specialized deep learning pipeline designed to classify an extensive range of crop types and crop diseases (65 distinct classes). The project utilizes a ResNet50-based architecture trained progressively in three distinct phases to ensure optimal feature extraction and high classification accuracy.

## Project Structure
- **`app.py`**: The main application entry point.
- **`src/`**: Core logic including vision, NLP, and pipeline scripts.
- **`data/class_names.json`**: Defines the 65 agricultural classes predicted by the model (e.g., Apple Scab, Corn Leaf Blight, Tomato Yellow Leaf Curl Virus, etc.).
- **`notebooks/cropquery.ipynb`**: The primary notebook encompassing the full pipeline, inference setup, and evaluation metrics. **All notebook usage is driven from this file.**
- **`models/phase3_final.pth`**: The final, fully-trained PyTorch model weights used for inference.
- **`notebooks/training/`**: Historical training files (`resnet50-phase-1-cropquery.ipynb`, `resnet50-phase-2-cropquery.ipynb`, etc.) used to progressively train the model.

## Setup & Dependencies

1. Ensure you have Python 3.8+ installed.
2. Install the necessary machine learning libraries:
   ```bash
   pip install torch torchvision numpy pandas matplotlib jupyter
   ```

## Usage Step-by-Step

Since the model is already trained, you only need the main notebook to evaluate it or run inferences.

1. **Open the Project Notebook**
   Open **`notebooks/cropquery.ipynb`** in your Jupyter environment or VS Code.

2. **Import Libraries and Config**
   Run the first few cells to import required libraries (`torch`, `torchvision`, etc.) and load mapping labels from `data/class_names.json`.

3. **Initialize the Model**
   Run the cells that define the ResNet50 architecture and load the saved optimal weights from `models/phase3_final.pth`.

4. **Run Inference & Evaluation**
   Execute the remaining inference cells in the notebook. This will:
   - Process your input agricultural imagery.
   - Run predictions using the loaded model.
   - Output the identified crop type and/or disease status directly in the notebook with visualizations.

## Supported Crops
The system detects health issues and states across a wide variety of flora, including but not limited to:
- Apples, Cherries, Peaches, Grapes, Strawberries
- Bell Peppers, Tomatoes, Potatoes, Pumpkins, Peas, Melons
- Corn, Rice, Wheat, Sugarcane
- Coffee, Cashew, Cassava
