import autokeras as ak

from tensorflow.keras.datasets import mnist

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28 * 28 * 1)
x_test = x_test.reshape(-1, 28 * 28 * 1)

print(x_train.shape, x_test.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

'''
# one hot incording
from tensorflow.keras.utiles import to_categorical
y_train = to_cateforical(y_train)
y_test = to_categorical(y_test)
'''

# 2. 모델
inputs = ak.ImageInput()
outputs = ak.ImageBlock(
    block_type='resnet',
    normalize=True,
    augment=False
)(inputs)
outputs = ak.ClassificationHead()(outputs)

model = ak.AutoModel(
    inputs = inputs, outputs= outputs, overwrite=True, max_trials=1
)

'''
model = ak.ImageClassifier(
    overwrite=True,
    max_trials=1
)
'''

# 3. 컴파일, 훈련
model.fit(x_train, y_train, epochs=2)

# 4. 평가, 예측
y_predict = model.predict(x_test)
print(y_predict[:10])

results = model.evaluate(x_test, y_test)
print(results)

model2 = model.export_model()
model2.summary()