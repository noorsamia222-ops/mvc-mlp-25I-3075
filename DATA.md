# Data

The MNIST dataset is automatically downloaded and preprocessed when the notebook runs. 

## Dataset Details
- **Training set:** 60,000 images
- **Test set:** 10,000 images
- **Dimensions:** 28x28 pixels (784 input features)
- **Labels:** Handwritten digits 0-9
- **Normalization:** Pixel values are scaled from 0-255 to 0-1 for Sigmoid stability.

## How it loads
The dataset is fetched automatically using the integrated Keras utility to ensure the environment is ready for the NumPy MLP implementation (Roll No: 25I-3075).

```python
from tensorflow.keras.datasets import mnist

# Load the data
(X_train_raw, Y_train_labels), (X_test_raw, Y_test_labels) = mnist.load_data()

# Data is then flattened and normalized within the notebook cell
