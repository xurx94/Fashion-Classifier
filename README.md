# PyTorch Fashion Classifier 🧥👟

An end-to-end Machine Learning pipeline built from scratch using PyTorch. This project trains a Multi-Layer Perceptron (MLP) to classify clothing items from the Fashion MNIST dataset.

**Final Test Accuracy:** 90.52% 🏆

## 🧠 Project Architecture

This is a modular machine learning ecosystem separated into functional components:

* `Fashion_MNIST_model.py`: The core PyTorch Neural Network architecture.
* `dataPreprocessing.py`: Handles raw CSV ingestion, Train/Validation/Test splitting, and pixel normalization (Z-scaling).
* `dataset.py`: Manages the PyTorch `DataLoader` batching and hardware routing (CPU/CUDA).
* `Model_Training.py`: The main engine. Features a custom training loop with validation-based checkpointing (saves the `state_dict` only when a new validation high score is achieved). It trains the model on around 48k images and validates on 12k images and saves the model only when the validation score is higher than before.
* `plotting.py`: A custom Matplotlib/Seaborn utility library for rendering image grids .
* `evaluate.py`: The bulk-testing script. Loads the saved model weights and evaluates them against the entire 10,000-image test dataset to calculate final mathematical accuracy..

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/Fashion-Classifier.git](https://github.com/your-username/PyTorch-Fashion-Classifier.git)
   cd Fashion-Classifier
```

2. **Install the required dependencies:**
```bash
pip install torch pandas numpy matplotlib seaborn pillow scikit-learn

```

3. **Download the Data (Required for Training):**
To keep this repository lightweight, the raw CSV datasets are not included.
* Go to the official [Zalando Fashion MNIST Kaggle Dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist).
* Download the archive and extract `fashion-mnist_train.csv` and `fashion-mnist_test.csv`.
* Place both CSV files in the root directory of this project.

## 🚀 How to Run

### 1. Train the Model

Once your data is placed in the root folder, initiate the training loop:

```bash
python Model_Training.py

```
*The script will automatically route to a CUDA GPU if available and generate a `Fashion_MNIST_TorchModel.pth` file.*

### 2. Run Saved Model on Test Dataset

You can test the saved model the network on test dataset.

```bash
python evaluate.py

```

## 🛠️ Built With

* **PyTorch** - Deep learning framework
* **Pandas & Scikit-Learn** - Data preprocessing and scaling
* **Matplotlib & Seaborn** - Diagnostic visualizations
