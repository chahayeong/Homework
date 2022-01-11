import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import os
import tqdm

from konlpy.tag import Okt

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score,f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

train=pd.read_csv('./dacon/_data02/train.csv')
test=pd.read_csv('./dacon/_data02/test.csv')
sample_submission=pd.read_csv('./dacon/_data02/sample_submission.csv')



# print(train.head(2))
'''
   index  제출년도  ...                                          요약문_영문키워드 label
0      0  2016  ...  nucleotide sequence, molecular marker, species...    24
1      1  2019  ...  TRAIL,Colorectal cancer,TRAIL resistance,Apopt...     0
'''

# print(test.head(2))
'''
    index  ...                                          요약문_영문키워드
0  174304  ...  Friction Stir Spot Welding, Non-destructive ev...
1  174305  ...  many particle system,stability of dynamics,qua...
'''

# print(sample_submission.head(6))
'''
    index  label
0  174304      0
1  174305      0
2  174306      0
3  174307      0
4  174308      0
5  174309      0
'''



train=train[['과제명','label']]
test=test[['과제명']]

#1. re.sub 한글 및 공백을 제외한 문자 제거
#2. okt 객체를 활용해 형태소 단위로 나눔
#3. remove_stopwords로 불용어 제거 
def preprocessing(text, okt, remove_stopwords=False, stop_words=[]):
    text=re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ]","", text)
    word_text=okt.morphs(text, stem=True)
    if remove_stopwords:
        word_review=[token for token in word_text if not token in stop_words]
    return word_review

stop_words=['은','는','이','가', '하','아','것','들','의','있','되','수','보','주','등','한']

okt=Okt()

clean_train_text=[]
clean_test_text=[]

#시간이 많이 걸립니다.
for text in tqdm.tqdm(train['과제명']):
    try:
        clean_train_text.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
    except:
        clean_train_text.append([])

for text in tqdm.tqdm(test['과제명']):
    if type(text) == str:
        clean_test_text.append(preprocessing(text, okt, remove_stopwords=True, stop_words=stop_words))
    else:
        clean_test_text.append([])

from sklearn.feature_extraction.text import CountVectorizer

#tokenizer 인자에는 list를 받아서 그대로 내보내는 함수를 넣어줍니다. 또한 소문자화를 하지 않도록 설정해야 에러가 나지 않습니다.
vectorizer = CountVectorizer(tokenizer = lambda x: x, lowercase=False)
train_features=vectorizer.fit_transform(clean_train_text)
test_features=vectorizer.transform(clean_test_text)
#test데이터에 fit_transform을 할 경우 data leakage에 해당합니다






# 2. 모델

#훈련 데이터 셋과 검증 데이터 셋으로 분리
TEST_SIZE=0.2
RANDOM_SEED=42

train_x, eval_x, train_y, eval_y=train_test_split(
    train_features, train['label'], test_size=TEST_SIZE, random_state=RANDOM_SEED)

#랜덤포레스트로 모델링
from sklearn.ensemble import RandomForestClassifier

forest=RandomForestClassifier(n_estimators=100)



# 3. 컴파일

forest.fit(train_x, train_y)

forest.score(eval_x, eval_y)



# 4. 평가

forest.predict(test_features)
sample_submission['label']=forest.predict(test_features)

sample_submission.to_csv('rf_baseline.csv', index=False)

