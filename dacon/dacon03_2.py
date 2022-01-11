#fingerfrint name to use
ffpp = "pattern"

#read csv
import pandas as pd
train = pd.read_csv("./dacon/_data03/train.csv")
dev = pd.read_csv("./dacon/_data03/dev.csv")
test = pd.read_csv("./dacon/_data03/test.csv")

ss = pd.read_csv("./dacon/_data03/sample_submission.csv")

train = pd.concat([train,dev])

train['ST1_GAP(eV)'] = train['S1_energy(eV)'] - train['T1_energy(eV)']

import kora.install.rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys

import numpy as np
import pandas as pd

import math
train_fps = []#train fingerprints
train_y = [] #train y(label)

for index, row in train.iterrows() : 
  try : 
    mol = Chem.MolFromSmiles(row['SMILES'])
    if ffpp == 'maccs' :    
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif ffpp == 'morgan' : 
        fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4)
    elif ffpp == 'rdkit' : 
        fp = Chem.RDKFingerprint(mol)
    elif ffpp == 'pattern' : 
        fp = Chem.rdmolops.PatternFingerprint(mol)
    elif ffpp == 'layerd' : 
        fp = Chem.rdmolops.LayeredFingerprint(mol)

    train_fps.append(fp)
    train_y.append(row['ST1_GAP(eV)'])
  except : 
    pass

#fingerfrint object to ndarray
np_train_fps = []
for fp in train_fps:
  arr = np.zeros((0,))
  DataStructs.ConvertToNumpyArray(fp, arr)
  np_train_fps.append(arr)

np_train_fps_array = np.array(np_train_fps)

pd.Series(np_train_fps_array[:,0]).value_counts()

import math
test_fps = []#test fingerprints
test_y = [] #test y(label)

for index, row in test.iterrows() : 
  try : 
    mol = Chem.MolFromSmiles(row['SMILES'])

    if ffpp == 'maccs' :    
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif ffpp == 'morgan' : 
        fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 4)
    elif ffpp == 'rdkit' : 
        fp = Chem.RDKFingerprint(mol)
    elif ffpp == 'pattern' : 
        fp = Chem.rdmolops.PatternFingerprint(mol)
    elif ffpp == 'layerd' : 
        fp = Chem.rdmolops.LayeredFingerprint(mol)

    test_fps.append(fp)
    test_y.append(row['ST1_GAP(eV)'])
  except : 
    pass

np_test_fps = []
for fp in test_fps:
  arr = np.zeros((0,))
  DataStructs.ConvertToNumpyArray(fp, arr)
  np_test_fps.append(arr)

np_test_fps_array = np.array(np_test_fps)

print(np_test_fps_array.shape)
# print(len(test_y))

pd.Series(np_test_fps_array[:,0]).value_counts()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
def create_deep_learning_model():
    model = Sequential()
    model.add(Dense(256, input_dim=2048, kernel_initializer='normal', activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics='acc')    
    return model

X, Y = np_train_fps_array , np.array(train_y)

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

estimators = []
# estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=create_deep_learning_model, epochs=10)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=5)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("%.2f (%.2f) MAE" % (results.mean(), results.std()))

model = create_deep_learning_model()
model.fit(X, Y, epochs = 100, batch_size=512, verbose=1,
    validation_split=0.25, callbacks=[es])
test_y = model.predict(np_test_fps_array)
ss['ST1_GAP(eV)'] = test_y

loss = model.evaluate(X, Y)
print('loss : ', loss[0])
print('acc : ', loss[1])

ss.to_csv("dacon03_06.csv",index=False)


'''
02
loss :  0.10273174941539764
acc :  0.00029658921994268894

03
loss :  0.11570936441421509
acc :  0.000263634865405038

04
loss :  0.0889880359172821
acc :  0.000263634865405038

05
loss :  0.128644660115242
acc :  0.00023068051086738706
'''