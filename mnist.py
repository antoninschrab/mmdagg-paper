"""
This code was extracted from https://github.com/MPI-IS/tests-wo-splitting under The MIT License.
Jonas M. Kübler, Wittawat Jitkrittum, Bernhard Schölkopf, Krikamol Muandet
Learning Kernel Tests Without Data Splitting
Neural Information Processing Systems 2020
https://papers.nips.cc/paper/2020/file/44f683a84163b3523afe57c2e008bc8c-Paper.pdf

The download_mnist function downloads the MNIST dataset 
and downsamples it to 7x7 images. The data mnist_7x7.data 
is the same as the considered by the above authors. It 
should be run only once. The load_mnist function loads
datasets consisting of images of various digits. 
"""

import pickle
from sklearn.datasets import fetch_openml
import numpy as np
from pathlib import Path


def download_mnist():
    """
    Download MNIST dataset and downsample it to 7x7 images,
    save the downsampled dataset as mnist_7x7.data in the
    mnist_dataset directory.
    """
    X, y = fetch_openml("mnist_784", return_X_y=True)
    X = X.to_numpy()
    X = X / 255
    digits = {}
    for i in range(10):
        digits[str(i)] = []
    for i in range(len(y)):
        digits[y[i]].append(X[i])
    digits_7x7 = {}
    for i in range(10):
        current = np.array(digits[str(i)])
        n = len(current)
        # make the dataset 2D again
        current = np.reshape(current, (n, 28, 28))
        current = np.reshape(current, (n, 7, 4, 7, 4))
        current = current.mean(axis=(2, 4))
        digits_7x7[str(i)] = np.reshape(current, (n, 49))
    path = "mnist_dataset/mnist_7x7.data"
    f = open(path, 'wb')
    pickle.dump(digits_7x7, f)
    f.close()


def load_mnist():
    """
    Returns P and Q_list where P consists of images of all digits 
    in mnist_7x7.data, and Q_list contains 5 elements each consisting
    of images of fewer digits.
    This function should only be run after download_mnist().
    """
    with open('mnist_dataset/mnist_7x7.data', 'rb') as handle:
        X = pickle.load(handle)
    P  = np.vstack(
        (X['0'], X['1'], X['2'], X['3'], X['4'], X['5'], X['6'], X['7'], X['8'], X['9'])
    )
    Q1 = np.vstack((X['1'], X['3'], X['5'], X['7'], X['9']))
    Q2 = np.vstack((X['0'], X['1'], X['3'], X['5'], X['7'], X['9']))
    Q3 = np.vstack((X['0'], X['1'], X['2'], X['3'], X['5'], X['7'], X['9']))
    Q4 = np.vstack((X['0'], X['1'], X['2'], X['3'], X['4'], X['5'], X['7'], X['9']))
    Q5 = np.vstack(
        (X['0'], X['1'], X['2'], X['3'], X['4'], X['5'], X['6'], X['7'], X['9'])
    )
    Q_list = [Q1, Q2, Q3, Q4, Q5]
    return P, Q_list


if __name__ == "__main__":
    Path("mnist_dataset").mkdir(exist_ok=True)
    if Path("mnist_dataset/mnist_7x7.data").is_file() == False:
        download_mnist()
