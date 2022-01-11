from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19

model = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
# model = VGG16()
# model = VGG19()

model.trainable=False

model.summary()

print(len(model.weights))
print(len(model.trainable_weights))



''' VGG16
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
'''


''' VGG19
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792 

'''''''''''''''''''''''''''''''
flatten (Flatten)            (None, 25088)             0
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312
_________________________________________________________________
predictions (Dense)          (None, 1000)              4097000
=================================================================
Total params: 143,667,240
Trainable params: 143,667,240
Non-trainable params: 0
'''


# FC 용어 정리
# CNN의 레이어로, 이미지를 분류/설명하는 데 가장 적합하게 예측
