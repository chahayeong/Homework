from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs = ['너무 재밌어요', '참 최고에요', '참 잘 만든 영화에요',
        '추천하고 싶은 영화입니다.', '한 번 더 보고 싶네요', '글쎄요',
        '별로에요', '생각보다 지루해요', '연기가 어색해요',
        '재미없어요', '너무 재미없다', '참 재밌네요', '청순이가 잘 생기긴 했어요']


# 긍정 1, 부정 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

x = token.texts_to_sequences(docs)
print(x)

from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5)   # post
print(pad_x)
print(pad_x.shape)

word_size = len(token.word_index)
print(word_size)

print(np.unique(pad_x))

# 원 핫 인코딩 하면? (13, 5) -> (13, 5, 27)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input

# 모델
'''
model = Sequential()                    # 인풋은 (13, 5)

# inpuit_dim = 단어사전의 개수, 라벨의 개수     input_length = 단어수, 길이
model.add(Embedding(input_dim=128, output_dim=77, input_length=5))
# model.add(Embedding(27,77))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
'''
# 함수형으로 고치기

# input1 = Input(shape = (5,))      # 아래와 동일한 파라미터 값
input1 = Input(shape = (None,))
em1 = Embedding(input_dim=28, output_dim=77)(input1)
em1 = LSTM(32)(em1)
output1 = Dense(1, activation='relu')(em1)

model = Model(inputs = input1, outputs = output1)


model.summary()

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, 5, 77)             2079
# _________________________________________________________________
# lstm (LSTM)                  (None, 32)                14080
# _________________________________________________________________
# dense (Dense)                (None, 1)                 33
# =================================================================
# Total params: 16,192
# Trainable params: 16,192
# Non-trainable params: 0

'''
# 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
                metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=32)

# 평가, 예측
acc = model.evaluate(pad_x, labels)[1]
print("acc :", acc)
'''