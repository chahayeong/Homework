from sklearn.datasets import load_iris, load_boston, load_breast_cancer, load_diabetes, load_wine
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100

iris_datasets = load_iris()
boston_datasets = load_boston()
cancer_datasets = load_breast_cancer()
diabetes_datasets = load_diabetes()
wine_datasets = load_wine()

x_data_iris = iris_datasets.data
y_data_iris = iris_datasets.target

x_data_boston = boston_datasets.data
y_data_boston = boston_datasets.target

x_data_cancer = cancer_datasets.data
y_data_cancer = cancer_datasets.target

x_data_diabetes = diabetes_datasets.data
y_data_diabetes = diabetes_datasets.target

x_data_wine = wine_datasets.data
y_data_wine = wine_datasets.target

#####################################

(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()

(x_train_fashion, y_train_fashion), (x_test_fashion, y_test_fashion)= fashion_mnist.load_data()

(x_train_cifar10, y_train_cifar10), (x_test_cifar10, y_test_cifar10) = cifar10.load_data()

(x_train_cifar100, y_train_cifar100), (x_test_cifar100, y_test_cifar100) = cifar100.load_data()




# print(type(x_data), type(y_data))

np.save('./_save/_npy/k55_iris_x_data.npy', arr=x_data_iris)
np.save('./_save/_npy/k55_iris_y_data.npy', arr=y_data_iris)

np.save('./_save/_npy/k55_boston_x_data.npy', arr=x_data_boston)
np.save('./_save/_npy/k55_boston_y_data.npy', arr=y_data_boston)

np.save('./_save/_npy/k55_cancer_x_data.npy', arr=x_data_cancer)
np.save('./_save/_npy/k55_cancer_y_data.npy', arr=y_data_cancer)

np.save('./_save/_npy/k55_diabetes_x_data.npy', arr=x_data_diabetes)
np.save('./_save/_npy/k55_diabetes_y_data.npy', arr=y_data_diabetes)

np.save('./_save/_npy/k55_wine_x_data.npy', arr=x_data_wine)
np.save('./_save/_npy/k55_wine_y_data.npy', arr=y_data_wine)

np.save('./_save/_npy/k55_mnist_x_data.npy', arr=x_train_mnist)
np.save('./_save/_npy/k55_mnist_y_data.npy', arr=y_train_mnist)

np.save('./_save/_npy/k55_fashion_x_data.npy', arr=x_train_fashion)
np.save('./_save/_npy/k55_fashion_y_data.npy', arr=y_train_fashion)

np.save('./_save/_npy/k55_cifar10_x_data.npy', arr=x_train_cifar10)
np.save('./_save/_npy/k55_cifar10_y_data.npy', arr=y_train_cifar10)

np.save('./_save/_npy/k55_cifar100_x_data.npy', arr=x_train_cifar100)
np.save('./_save/_npy/k55_cifar100_y_data.npy', arr=y_train_cifar100)