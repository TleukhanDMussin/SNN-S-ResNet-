# Spiking Neural Network (SNN) for DVS Gesture Recognition

## Overview
This project implements a Spiking Neural Network (SNN) based on a ResNet-inspired architecture, specifically designed to process Dynamic Vision Sensor (DVS) data for gesture recognition tasks. The SNN integrates spiking neuron dynamics and approximates spiking behaviors using custom membrane potential updates and activation functions.

---

## Files in the Repository

### 1. `SResNet.py`
This file defines the architecture of the SNN model, including:
- **SpikingBasicBlock**: A custom block integrating convolutional layers, membrane potential updates, and spiking dynamics.
- **SpikingResNet**: The overall ResNet-like structure adapted for SNNs.
- **Key Components**:
  - **Membrane Potential Update (`mem_update`)**: Implements the spiking neuron dynamics with a decay factor and firing threshold.
  - **Custom Activation Function (`ActFun`)**: Approximates the spiking behavior with a smooth gradient for backpropagation.

### 2. `Test_SNN.py`
This script is responsible for:
- **Data Loading**: Using the `tonic` library to load and preprocess the DVS Gesture dataset.
- **Training**:
  - Utilizes cross-entropy loss and SGD optimizer.
  - Implements a learning rate scheduler for gradual decay.
- **Validation**:
  - Evaluates the model's performance after every training epoch.
- **Checkpointing**:
  - Saves the best-performing model based on validation accuracy.

### 3. `accuracy.py`
Utility functions for:
- **Accuracy Calculation**: Computes top-1 and top-5 accuracy metrics.
- **Logging**: Records training and validation metrics.
- **Checkpoint Management**: Saves and loads model checkpoints during training.

---

## Dataset
The project uses the **DVS Gesture Dataset**, which consists of:
- Event-based data collected from a dynamic vision sensor.
- Input format: `(x, y, polarity, timestamp)`.

### Preprocessing
- **Transformations Applied**:
  - **Denoise**: Removes noisy events based on a temporal threshold.
  - **Frame Conversion**: Converts event streams into frame-based representations for input into the SNN.

---

## Model Architecture
The SpikingResNet architecture is based on ResNet, with modifications to support spiking neuron dynamics.

### Key Features:
- **SpikingBasicBlock**: Incorporates membrane potential updates and spiking activation functions.
- **Custom Layers**:
  - Dropout for regularization.
  - Average pooling for down-sampling.
- **Membrane Potential Dynamics**:
  - Decay constant: `0.8` (configurable).
  - Threshold: `0.5` for spiking activation.

---

## Training Details
### Hyperparameters:
- Learning Rate: `0.01` (with decay every 20 epochs).
- Batch Size: `16`.
- Number of Epochs: `50`.
- Optimizer: Stochastic Gradient Descent (SGD).
- Loss Function: Cross-Entropy Loss.

### Training Process:
1. Spiking dynamics are incorporated into each forward pass.
2. Gradients are computed using a smooth approximation of the spiking function.
3. The model is trained to classify gestures into one of 11 classes.

---

## Validation
During validation:
- The model's predictions are compared against ground truth labels.
- Accuracy is calculated using the top-1 and top-5 metrics.

---

## How to Run
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision tonic matplotlib tqdm
   ```

3. Prepare the dataset:
   The DVS Gesture dataset will be automatically downloaded and processed by the `tonic` library.

4. Train the model:
   ```bash
   python Test_SNN.py
   ```

5. Check validation performance and view saved checkpoints:
   - The best model checkpoint is saved as `model_best.pth.tar`.

---

## Suggested Improvements
- **Hyperparameter Tuning**: Experiment with different decay constants, dropout rates, and learning rates.
- **Data Augmentation**: Introduce temporal jittering or noise injection to make the model more robust.
- **Advanced Architectures**: Use deeper networks or adapt modern spiking architectures for better performance.
- **Mixed Precision Training**: Implement AMP (Automatic Mixed Precision) to optimize training time and memory usage.

---

## Acknowledgments
- The architecture is inspired by ResNet and adapted for spiking dynamics.
- The DVS Gesture dataset is provided by the **tonic** library.
- Frameworks used: PyTorch, Tonic, and Matplotlib.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Contact
For questions or contributions, please reach out to the project maintainer.

