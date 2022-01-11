import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers  import Dense, Dropout, Input, Conv2D

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255


# 2. 모델
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28*28), name='input')
    x = Dense(512, activation='relu', name='hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu', name='hidden2')(inputs)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='hidden3')(inputs)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],
                    loss='categorical_crossentropy')
    return model

def create_hyperparameter():
    batches = [100, 200] #, 30, 40, 50]
    optimizers = ['rmsprop'] # , 'adam', 'adadelta']
    dropout = [0.1] # , 0.2, 0.3]
    return {"batch_size" : batches, "optimizer" : optimizers,
            "drop" : dropout}

hyperparameters = create_hyperparameter()
# print(hyperparameters)


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# model2 = build_model()        # 아래처럼 써준다
model2 = KerasClassifier(build_fn=build_model, verbose=1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# model = RandomizedSearchCV(model2, hyperparameters, cv=5)
model = GridSearchCV(model2, hyperparameters, cv=2)

model.fit(x_train, y_train, verbose=1, epochs=3) #, validation_split=0.2)

print(model.best_params_)
print(model.best_estimator_)
print(model.best_score_)
acc = model.score(x_test, y_test)
print("최종 스코어 : ", acc)


'''
최종 스코어 :  0.9696000218391418
'''














