import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from Fashion_MNIST_modelCNN import dynamiCNN


def predict_gallery_photo(image_path, model_weight_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelx = dynamiCNN(1,28,3,[32,64,128],2,[256,256],10,0.21).to(device)
    
    label_map = {
        0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
        5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
    }

    try:
        modelx.load_state_dict(torch.load(model_weight_path, map_location=device))
        print("Model weights loaded successfully!")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    modelx.to(device)
    modelx.eval()

    try:
        raw_img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Could not find the image at {image_path}")
        return

    img = raw_img.convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28, 28))

    img_array = np.array(img) / 255.0
    img_tensor = torch.tensor(img_array, dtype=torch.float32).reshape(-1, 1, 28, 28).to(device)

    with torch.no_grad():
        raw_logits = modelx(img_tensor)
        probabilities = torch.nn.functional.softmax(raw_logits, dim=1)[0]
        confidence_score, prediction_index = torch.max(probabilities, 0)

    predicted_label = label_map[prediction_index.item()]
    confidence_percent = confidence_score.item() * 100

    plt.figure(figsize=(5, 5))
    plt.imshow(img_array, cmap='gray')
    plt.title(f"Model Predicts: {predicted_label} \n Model's {confidence_percent:.1f}% Damn sure it's a {predicted_label}", fontsize=12)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':

    MY_PHOTO = "filename.jpg" 
    MY_SAVED_MODEL = "Fashion-mnistCNN.pth"

    predict_gallery_photo(MY_PHOTO, MY_SAVED_MODEL)
