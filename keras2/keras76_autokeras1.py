# cmd
# pip install autokeras

import autokeras as ak

from tensorflow.keras.datasets import mnist

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 모델
model = ak.ImageClassifier(
    overwrite=True,
    max_trials=2
)



# 3. 컴파일, 훈련
model.fit(x_train, y_train, epochs=5, validation_split=0.15)

# 4. 평가, 예측
y_predict = model.predict(x_test)

results = model.evaluate(x_test, y_test)
print(results)
