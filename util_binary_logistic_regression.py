from numpy import exp
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def toy_data(M=10000, train_size=0.8):
    # M number of samples per feature
    #train_size = 0.8  # 80% of data are used for training, 20% for testing
    N = 2  # number of features (excluding bias)
    random_state = 12345
    X, Y = make_classification(
        n_samples=M,
        n_features=N,
        n_informative=N,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=1,
        flip_y=0.01,
        random_state=random_state,
    )
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=train_size, random_state=random_state
    )
    return X_train, Y_train, X_test, Y_test

def init_weights():
    # nice numbers here, in practice these
    # are appropriately randomly sampled numbers
    w1 = 0.5
    w2 = 0.25
    b = 0.125
    return w1, w2, b

def my_sigmoid(z):
    return 1.0 / (1.0 + exp(-z))

def predict_class(y):
    return (y >=0.5) * 1