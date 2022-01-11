from sklearn.utils import validation
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import numpy as np

datasets = load_iris()

x = datasets.data 
y = datasets.target 

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.1, shuffle=True, random_state=12)

x_train = x_train.reshape(x_train.shape[0],4,1).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0],4,1).astype('float32')/255

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta

def build_model(drop=0.5, node=2, activation='relu', learning_rate=0.001):
    inputs = Input(shape=(4,1))
    x = Conv1D(filters = 128, kernel_size=2, activation= activation)(inputs)
    x = Dropout(drop) (x)
    x = MaxPooling1D(2,2) (x)
    x = Flatten() (x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate), metrics=['acc'],
                    loss='categorical_crossentropy')
    return model                    

def create_hyperparameter():
    batches = [100,200]
    optimizer = [Adam, RMSprop, Adadelta]
    activation = ['selu','relu']
    dropout = [0.3, 0.5]
    node = [64, 128]
    epochs = [10, 20]
    validation_split = [0.15, 0.2]
    learning_rate = [0.001, 0.01]
    return {"batch_size":batches, "drop":dropout, "activation":activation, "node":node, "epochs":epochs,
            "validation_split":validation_split, "learning_rate":learning_rate     
    }

hyperparameter = create_hyperparameter()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

es = EarlyStopping(monitor = 'val_loss',patience=10)

model2 = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

model = RandomizedSearchCV(model2, hyperparameter, cv=2)

model.fit(x_train, y_train, verbose=1, callbacks = [es])


print(model.best_params_)
print(model.best_estimator_)
print(model.best_score_)
acc = model.score(x_train, y_train)
print('최종 스코어 :', acc)


'''
최종 스코어 : 0.3481481373310089
'''