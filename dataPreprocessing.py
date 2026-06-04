import pandas as pd
from sklearn.model_selection import train_test_split 

df = pd.read_csv('fashion-mnist_train.csv')
dfT = pd.read_csv('fashion-mnist_test.csv')

df.dropna(inplace=True)
dfT.dropna(inplace=True) 

def preprocessedData():
    x_raw = df.iloc[:, 1:].to_numpy()
    y_raw = df.iloc[:, 0].to_numpy()

    x_train, x_valid, y_train, y_valid = train_test_split(x_raw, y_raw, test_size=0.2, random_state=42)

    x_test = dfT.iloc[:, 1:].to_numpy()
    y_test = dfT.iloc[:, 0].to_numpy()

    x_train = x_train / 255.0
    x_valid = x_valid / 255.0 
    x_test = x_test / 255.0

    return x_train, y_train, x_valid, y_valid, x_test, y_test