# Crop-Query

A Multilingual Vision-Language System for Intelligent Agricultural Monitoring.

## Overview
Crop-Query is a specialized deep learning pipeline designed to classify an extensive range of crop types and crop diseases (65 distinct classes). The project utilizes a ResNet50-based architecture trained progressively in three distinct phases to ensure optimal feature extraction and high classification accuracy.

## Project Structure
- **`cropquery.ipynb`**: The primary notebook encompassing the full pipeline, inference setup, and evaluation metrics.
- **`resnet50-phase-1-cropquery.ipynb`**: Training notebook for Phase 1 (Warm-up / Initial base training).
- **`resnet50-phase-2-cropquery.ipynb`**: Training notebook for Phase 2 (Intermediate fine-tuning).
- **`resnet50-phase-3-cropquery.ipynb`**: Training notebook for Phase 3 (Final optimization and stabilization).
- **`class_names.json`**: Defines the 65 agricultural classes predicted by the model (e.g., Apple Scab, Corn Leaf Blight, Tomato Yellow Leaf Curl Virus, etc.).
- **`*.pth` files**: Saved PyTorch weights corresponding to the best runs from each training phase (`phase1_best.pth`, `phase2_best.pth`, `phase3_best.pth`, `phase3_final.pth`).

## Setup & Dependencies

1. Ensure you have Python 3.8+ installed.
2. Install the necessary machine learning libraries (PyTorch, Torchvision, etc.):
   ```bash
   pip install torch torchvision numpy pandas matplotlib jupyter
   ```

## Usage Step-by-Step

### Step 1: Model Training (Optional, if starting from scratch)
The training requires executing three consecutive phases. Each phase builds upon the weights of the previous one.
1. **Phase 1**: Open and run `resnet50-phase-1-cropquery.ipynb`. This outputs `phase1_best.pth`.
2. **Phase 2**: Open and run `resnet50-phase-2-cropquery.ipynb`. It will load Phase 1 weights and output `phase2_best.pth`.
3. **Phase 3**: Open and run `resnet50-phase-3-cropquery.ipynb`. It will load Phase 2 weights and output `phase3_best.pth` and `phase3_final.pth`.

### Step 2: Evaluation and Inference
To run inferences on images or evaluate the trained model:
1. Open the main **`cropquery.ipynb`** notebook.
2. Ensure the `class_names.json` and the final model weights (`phase3_final.pth` or `phase3_best.pth`) are in the root directory.
3. Run the inference cells to load the model architecture, apply the saved weights, and perform classification tasks on your agricultural imagery.

## Supported Crops
The system detects health issues and states across a wide variety of flora, including but not limited to:
- Apples, Cherries, Peaches, Grapes, Strawberries
- Bell Peppers, Tomatoes, Potatoes, Pumpkins, Peas, Melons
- Corn, Rice, Wheat, Sugarcane
- Coffee, Cashew, Cassava
