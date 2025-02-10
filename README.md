# Deep Learning Basics for Image Classification 

The focus is on building a linear classifier and a Convolutional Neural Network (CNN) for image classification tasks, particularly using the CIFAR-10 dataset.The dataset consists of 60,000 32x32 color images across 10 different classes, with 50,000 images for training and 10,000 images for testing.

## Files in the Repository

- `Linear_classifier_FromScratch.ipynb`: Implements a basic linear classifier.
- `Linear_classifier_pytorch.ipynb`: Implements a linear classifier using PyTorch.
- `CNN.ipynb`: Implements a Convolutional Neural Network (CNN) using PyTorch.

## Dataset

The CIFAR-10 dataset is used for training and evaluation. The repository includes code to check if the dataset exists and downloads it if necessary.

## Features
- Implementation of Linear Classifier and CNN in PyTorch.
- Training and evaluation on CIFAR-10 dataset.
- Configurable hyperparameters via `CNN_config.yaml`.
- Dataset download and preprocessing included.
- Model training with GPU acceleration support.

## Model Architectures
### Linear Classifier
- Fully connected layers
- Softmax activation for classification
- Cross-entropy loss function

### Convolutional Neural Network (CNN)
- Multiple convolutional layers with ReLU activation
- Max pooling layers for down-sampling
- Fully connected layers for classification
- Softmax activation for output probabilities

## Prerequisites

- Python 3.x
- PyTorch
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/NusRAT-LiA/ML-ImageClassifier-FromScratch.git
   cd ML-ImageClassifier-FromScratch
   ```
2. Install dependencies:
   ```bash
   pip install torch torchvision numpy matplotlib
   ```

## Running the Notebooks

You can run the Jupyter notebooks using:
```bash
jupyter notebook
```
Open the required notebook and execute the cells step by step.

## Usage

- Modify hyperparameters in the respective notebooks.
- Train models on CIFAR-10.
- Evaluate performance and analyze results.

## Contributing
Feel free to submit issues or pull requests for improvements.

## Future Improvements
- Implement data augmentation techniques to improve model performance.
- Experiment with different architectures like ResNet and VGG.
- Add hyperparameter tuning and automated logging.
- Improve model efficiency with quantization or pruning.
- Implement real-time inference with a web or mobile application.

## License
This project is licensed under the MIT License.

## Acknowledgments
- PyTorch for deep learning framework.
- CIFAR-10 dataset for benchmarking image classification models.

