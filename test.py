import jieba.posseg as pseg
import jieba
import pandas as pd
import numpy as np
import keras
# 需要裝Tensorflow
from keras.layers import Embedding, Dense
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
# listall = cut_all(train_df)
X = []
tok = keras.preprocessing.text.Tokenizer(10000)
max_length = 600
for i in range(0,1000):

    listrow = cut_row(train_df,i)
    tok.fit_on_texts(listrow)
    arr = tok.texts_to_sequences(listrow)
    # arr = [one_hot(d, 1000) for d in listrow]
    # #刪除空白字詞
    arr = [item for sublist in arr for item in sublist]
    # arr = np.array(arr)
    # print(arr)
    X.append(arr)

X = pad_sequences(X, maxlen=max_length, padding='post')
print(X)

xtrain = X



# print(arr)

# for seq in arr:
#     print([tok.index_word[idx] for idx in seq])
# 確認轉換成功


ytrain = train_df['Final_Label_Kaggle']
ytrain = ytrain.str.split(' ') 

Y =[]
for i in range(0,1000):
   ytrain = tup(ytrain,i)
   
# for i in range(0,100):
#     Y.append(ytrain[i][1])

Y = ytrain[0:1000]
Yt = []
for i in range(0,1000):
    k = Y[i]
    Yt.append(k)
   
ytrain = np.array(Yt,dtype=np.float)
print(ytrain)

model = Sequential()
model.add(Embedding(10000,32,input_length=max_length))
model.add(Flatten())
model.add(Dense(512,input_dim=max_length,kernel_initializer='random_uniform', activation='relu'))
# model.add(layers.Dropout(0.2))
model.add(Dense(8, activation='relu'))
print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(xtrain, ytrain, epochs=10, batch_size=5, verbose=1)