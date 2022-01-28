import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, LSTM, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
import datetime
import random
import torch
import time



# 1. 데이터

train      = pd.read_csv("./dacon/_data/train_data.csv")
test       = pd.read_csv("./dacon/_data/test_data.csv")
submission = pd.read_csv("./dacon/_data/sample_submission.csv")
topic_dict = pd.read_csv("./dacon/_data/topic_dict.csv")

# train (45654, 3)
# test (9131, 2)

''' train
       index                               title  topic_idx
0          0            인천→핀란드 항공기 결항…휴가철 여행객 분통          4
1          1      실리콘밸리 넘어서겠다…구글 15조원 들여 美전역 거점화          4
2          2      이란 외무 긴장완화 해결책은 미국이 경제전쟁 멈추는 것          4
3          3    NYT 클린턴 측근韓기업 특수관계 조명…공과 사 맞물려종합          4
4          4           시진핑 트럼프에 중미 무역협상 조속 타결 희망          4
...      ...                                 ...        ...
45649  45649        KB금융 미국 IB 스티펠과 제휴…선진국 시장 공략          1
45650  45650     1보 서울시교육청 신종코로나 확산에 개학 연기·휴업 검토          2
45651  45651         게시판 키움증권 2020 키움 영웅전 실전투자대회          1
45652  45652                   답변하는 배기동 국립중앙박물관장          2
45653  45653  2020 한국인터넷기자상 시상식 내달 1일 개최…특별상 김성후          2
'''

''' test
      index                            title
0     45654       유튜브 내달 2일까지 크리에이터 지원 공간 운영     
1     45655          어버이날 맑다가 흐려져…남부지방 옅은 황사      
2     45656      내년부터 국가RD 평가 때 논문건수는 반영 않는다     
3     45657  김명자 신임 과총 회장 원로와 젊은 과학자 지혜 모을 것  
4     45658   회색인간 작가 김동식 양심고백 등 새 소설집 2권 출간   
...     ...                              ...
9126  54780     인천 오후 3시35분 대설주의보…눈 3.1cm 쌓여
9127  54781    노래방에서 지인 성추행 외교부 사무관 불구속 입건종합 
9128  54782     40년 전 부마항쟁 부산 시위 사진 2점 최초 공개       
9129  54783    게시판 아리랑TV 아프리카개발은행 총회 개회식 생중계  
9130  54784  유영민 과기장관 강소특구는 지역 혁신의 중심…지원책 강구
'''



# 1-1 전처리

def clean_text(sent):
    sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", sent)
    return sent_clean

train["cleaned_title"] = train["title"].apply(lambda x : clean_text(x))
test["cleaned_title"]  = test["title"].apply(lambda x : clean_text(x))

train_text = train["cleaned_title"].tolist()
test_text = test["cleaned_title"].tolist()
train_label = np.asarray(train.topic_idx)

tfidf = TfidfVectorizer(analyzer='word', sublinear_tf=True, ngram_range=(1, 2), max_features=150000, binary=False)

tfidf.fit(train_text)

train_tf_text = tfidf.transform(train_text).astype('float32')
test_tf_text  = tfidf.transform(test_text).astype('float32')


# 2. 모델
'''
model = Sequential()
model.add(Dense(128, input_dim = 32, activation = "relu"))
model.add(Dropout(0.8))
model.add(Dense(7, activation = "softmax"))
'''

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input, Conv1D, GlobalAveragePooling1D

model = Sequential()
model.add(Dense(32, input_dim = 150000, activation = "relu"))
model.add(Dropout(0.8))
model.add(Dense(7, activation = "softmax"))



# 3. 컴파일, 훈련
'''
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = tf.optimizers.Adam(0.001), metrics = ['accuracy'])

history = model.fit(x = train_inputs[:40000], y = labels[:40000],
                    validation_data =(train_inputs[40000:], labels[40000:]),
                    epochs = 100)
'''

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.Adam(0.001)
                , metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)



model.fit(train_tf_text, train_label, epochs=500, batch_size=128, verbose=1,
    callbacks=[es])






'''
# 4. 평가, 예측

tmp_pred = model.predict(test_inputs)
pred = np.argmax(tmp_pred, axis = 1)

submission.topic_idx = pred
submission.sample(3)

loss = model.evaluate(x = train_inputs[:40000], y = train_labels[:40000])

print('loss = ', loss[0])
print('acc : ', loss[1])
submission.to_csv('subfile_test.csv', index = False)
'''

tmp_pred = model.predict(test_tf_text)
pred = np.argmax(tmp_pred, axis = 1)

submission.topic_idx = pred
submission.sample(3)

loss = model.evaluate(train_tf_text, train_label)
print('loss : ', loss[0])
print('acc : ', loss[1])

submission.to_csv('subfile_07.csv', index = False)


'''
test1 0.786637
loss =  0.009336734190583229
acc :  0.9973000288009644

test2 0.757502
loss =  0.0019870756659656763
acc :  0.9993249773979187

test3 0.739978
loss =  0.004066706169396639
acc :  0.9982749819755554

test4 0.603285
loss =  0.2545240521430969
acc :  0.8874499797821045

test5 0.720481
loss =  0.0021422409918159246
acc :  0.9991750121116638

test6 0.716100
loss =  0.0104008624330163
acc :  0.9979749917984009

test7 0.6889375
loss =  0.02823515050113201
acc :  0.9961000084877014

test8 0.704052
loss =  0.015867039561271667
acc :  0.9967749714851379
'''

'''
01
loss :  0.0019176292698830366
acc :  0.999189555644989

02
loss :  0.001986767863854766
acc :  0.999189555644989

03
loss :  0.0020455694757401943
acc :  0.9992333650588989

04
loss :  0.0013433769345283508
acc :  0.9992552399635315

05
loss :  0.0015089423395693302
acc :  0.9992552399635315

06
loss :  0.012937832623720169
acc :  0.9988390803337097

06
loss :  0.012937832623720169
acc :  0.9988390803337097
'''