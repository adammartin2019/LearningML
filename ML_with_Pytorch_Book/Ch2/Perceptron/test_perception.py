import pandas as pd
import numpy as np
from perceptron import Perceptron


if __name__ == "__main__":
    data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    df = pd.read_csv(data, header=None, encoding='utf-8')
    
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)
    
    X = df.iloc[0:100, [0,2]].values
    
    ppn = Perceptron(eta=0.01, n_iter=10)
    ppn.fit(X, y)
    
    print("Errors:", ppn.errors_)
    

