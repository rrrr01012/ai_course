import jieba.posseg as pseg
import jieba
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
import csv
# 需要裝Tensorflow
from keras.layers import Embedding, Dense , LSTM , Dropout
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Flatten
from tensorflow.keras import models, layers 

def load_df(dataset_path): #讀資料
    df = pd.read_excel(dataset_path, sheet_name=0, header=0,
                                 converters={'id': str, 'name': str, 'description': str, 'category': str,
                                             'owner name': str, 'location': str})
    
    df['name'] = df['name'].fillna('')
    df['description'] = df['description'].fillna('')
    df['category'] = df['category'].fillna('')
    df['owner name'] = df['owner name'].fillna('')
    df['location'] = df['location'].fillna('')
    # 刪掉全部的NAN
    
    return df
def cut_row(df,i):
    texts = []
    for index, row in df.iterrows():
        text = row['name'] + ' ' + row['description'] + ' ' + row['category'] + ' ' + row['owner name'] + ' ' + row[
            'location']
        texts.append(text)
    texts[i] = texts[i].replace('\n','')
    texts[i] = texts[i].replace(' ','')
    texts[i] = texts[i].replace('(','')
    texts[i] = texts[i].replace(')','')
    texts[i] = texts[i].replace('：','')
    texts[i] = texts[i].replace('？','')
    texts[i] = texts[i].replace('~','')
    texts[i] = texts[i].replace('，','')
    texts[i] = texts[i].replace('）','')
    texts[i] = texts[i].replace('!','')
    texts[i] = texts[i].replace('】','')
    texts[i] = texts[i].replace('【','')
    
    # words = pseg.cut(texts[i])      
    # word = [w for w,f in words]
    word =' '.join(jieba.cut(texts[i],cut_all=False))
    # print(texts[i])
    # print(word)
    return word


def tup(ytrain,i):
    mlb = MultiLabelBinarizer(('0','1','2','3','4','5','6','7'))
    ytrain[i] = tuple(ytrain[i])
    ytrain[i] = mlb.fit_transform([ytrain[i]])
    ytrain[i] = ytrain[i][0]
    return ytrain
    


train_dataset_path = 'TrainingData.xlsx'
train_df = load_df(train_dataset_path)
test_dataset_path = 'TestingData.xlsx'
test_df = load_df(test_dataset_path)
# listall = cut_all(train_df)
X = []
X2 = []
tok = keras.preprocessing.text.Tokenizer(100000)
max_length = 800
# train_df.shape[0]
for i in range(0,train_df.shape[0]):

    listrow = cut_row(train_df,i)

    X.append(listrow)
tok.fit_on_texts(X)
arr = tok.texts_to_sequences(X)
xtrain = pad_sequences(arr, maxlen=max_length, padding='post')

for i in range(0,test_df.shape[0]):

    listrow = cut_row(test_df,i)

    X2.append(listrow)
tok.fit_on_texts(X2)
arr2 = tok.texts_to_sequences(X2)
xtest = pad_sequences(arr2, maxlen=max_length, padding='post')

ytrain = train_df['Final_Label_Kaggle']
ytrain = ytrain.str.split(' ') 

Y =[]
for i in range(0,1000):
   ytrain = tup(ytrain,i)


Y = ytrain[0:1000]
Yt = []
for i in range(0,1000):
    k = Y[i]
    Yt.append(k)
   
ytrain = np.array(Yt,dtype=np.float)


model = Sequential()
model.add(Embedding(100000,100,input_length=max_length))
model.add(Flatten())
# model.add(LSTM(64, return_sequences=True))
# model.add(Dropout(0.5))
# model.add(LSTM(32))
# model.add(Dropout(0.5))
model.add(Dense(800, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(800, activation='relu'))
model.add(Dense(8, activation='sigmoid'))
print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(xtrain, ytrain, epochs=10, batch_size=50, verbose=1)
ytest = model.predict(xtest)
print(ytest)


c=[]
e=[]
for i in range(0,len(ytest)):
    b=[]
    for j in range(0,8):
        if(ytest[i][j]>0.8):
            b.append(1)
        else:
            b.append(0)
    c.append(b)   
for i in range(0,len(c)):
    d=[]
    for j in range(0,8):
        if(c[i][j] == 1):
            d.append(j)
    e.append(d)
print(e[0])

def writecsv(result):
    # 寫檔
    with open('submission.csv', 'w', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)

        # 寫入第一列資料
        writer.writerow(['Id', 'Predicted'])
        for index, row in test_df.iterrows():
            ans = " ".join(str(i) for i in result[index])
            writer.writerow([test_df['id'][index],ans])
writecsv(e)