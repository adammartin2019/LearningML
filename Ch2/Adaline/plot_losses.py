import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adaline import AdalineGD

if __name__ == "__main__":
    data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    df = pd.read_csv(data, header=None, encoding='utf-8')

    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', 0, 1)

    X = df.iloc[0:100, [0,2]].values
    
    #plot data
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
    
    ada1 = AdalineGD(n_iter=15, eta=.1).fit(X,y)
    ax[0].plot(range(1, len(ada1.losses_) + 1),
               np.log10(ada1.losses_), marker='o')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('log(Mean squared error)')
    ax[0].set_title('Adaline - Learning rate .1')
    
    ada2 = AdalineGD(n_iter=15, eta=.0001).fit(X,y)
    ax[1].plot(range(1, len(ada2.losses_) + 1),
               ada2.losses_, marker='o')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('log(Mean squared error)')
    ax[1].set_title('Adaline - Learning rate .0001')
    
    plt.show()
    