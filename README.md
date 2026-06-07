# PyTorch Fashion Classifier 🧥👟

An end-to-end Machine Learning pipeline built from scratch using PyTorch. This project trains and evaluates neural network architectures to classify clothing items from the Fashion MNIST dataset, showing the progression from a basic Artificial Neural Network (ANN) to a more advanced Convolutional Neural Network (CNN).

**Final Test Accuracies:** 
* **CNN Model:** 93.08% 🏆
* **ANN Model:** 90.52%

## 🧠 Project Architecture

This is a modular machine learning ecosystem separated into functional components. 

* `Fashion_MNIST_model.py`: The original basic PyTorch ANN architecture.
* `Fashion_MNIST_modelCNN.py`: The upgraded PyTorch CNN architecture with batch normalization.
* `dataPreprocessing.py`: Handles raw CSV ingestion, Train/Validation/Test splitting, and pixel normalization (Z-scaling).
* `dataset.py`: Manages the PyTorch `DataLoader` batching and hardware routing (CPU/CUDA).
* `Model_Training.py`: The main engine. Features a custom training loop with validation-based checkpointing. It trains the model on around 48k images, validates on 12k images, and saves the weights (`.pth` file) only when a new validation high score is achieved.
* `plotting.py`: A custom Matplotlib/Seaborn utility library for rendering image grids.
* `evaluate.py`: The bulk-testing script for the ANN model. 
* `evaluateCNN.py`: The bulk-testing script for the new CNN model. Loads the saved CNN weights and evaluates them against the entire 10,000-image test dataset.
* `custom_photo_verificationCNN.py`: PIL Image preprocessing for custom photo classification.
* `evaluateCNN.py`: classifying a custom image from gallary

## ⚙️ Installation & Setup

1. **Clone the repository:**
```bash
   git clone [https://github.com/your-username/Fashion-Classifier.git](https://github.com/your-username/Fashion-Classifier.git)
   cd Fashion-Classifier
```

2. **Install the required dependencies:**

```bash
   pip install torch pandas numpy matplotlib seaborn pillow scikit-learn pillow
```

3. **Download the Data (Required for Training):**
To keep this repository lightweight, the raw CSV datasets are not included.
* Go to the official [Zalando Fashion MNIST Kaggle Dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist).
* Download the archive and extract `fashion-mnist_train.csv` and `fashion-mnist_test.csv`.
* Place both CSV files in the root directory of this project.


## 🚀 How to Run

### 1. Train the Model

Once your data is placed in the root folder, initiate the training loop. You can modify the import inside this file to train either the ANN:

```bash
python Model_Training.py
```
*The script will automatically route to a CUDA GPU if available and generate a `.pth` file containing the trained weights.*

for the CNN:
```bash
python Fashion_MNIST_modelCNN.py
```

### 2. Run Saved Models on Test Dataset

To verify the overall accuracy against the entire 10,000-image test dataset, run the evaluation script that matches your model.

For the CNN:

```bash
python evaluateCNN.py

```

For the ANN:

```bash
python evaluate.py

```

*These scripts will load the saved `.pth` files, process the test batches without updating gradients, and output the final percentage score.*

### 3. Classify custom images

To verify the model's strength, use custom images to see the models accuracy with real world data.


```bash
python evaluateCustomPhoto.py

```


## 🛠️ Built With

* **PyTorch** - Deep learning framework
* **Pandas & Scikit-Learn** - Data preprocessing and scaling
* **Matplotlib & Seaborn** - Diagnostic visualizations

```

```
