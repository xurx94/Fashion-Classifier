import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


sns.set_theme(style="whitegrid", palette="muted")

def plot_image_grid(x, y, df, img_size=(28, 28), label_col=0):

    fig, axes = plt.subplots(x, y, figsize=(max(x*2.5, 4), max(y*2.5, 4)), squeeze=False)
    fig.suptitle(f"Grayscaled Images ({x}x{y})", fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        if i >= len(df): 
            ax.axis('off')
            continue
            
        pixel_data = df.iloc[i].drop(df.columns[label_col]).to_numpy()
        img = pixel_data.reshape(img_size)
        raw_label = df.iloc[i, label_col]
        
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f"Label: {raw_label}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("fashion-mnist_train.csv")
    df=df.dropna()
    plot_image_grid(4,4,df)
