
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
print(test_df['name'][1000])

def cut_row(df,i):
    texts = []
    for index, row in df.iterrows():
        text = row['name'] + ' ' + row['description'] + ' ' + row['category'] + ' ' + row['owner name'] + ' ' + row[
            'location']
        texts.append(text)

    #手動排除沒用符號ㄏ    
    texts[i] = texts[i].replace('\n','')
    texts[i] = texts[i].replace(' ','')
    texts[i] = texts[i].replace('(','')
    texts[i] = texts[i].replace(')','')
    texts[i] = texts[i].replace('：','')
    texts[i] = texts[i].replace('？','')
    texts[i] = texts[i].replace('~','')
    texts[i] = texts[i].replace('，','')
    texts[i] = texts[i].replace('）','')
    texts[i] = texts[i].replace('（','')
    texts[i] = texts[i].replace('!','')
    texts[i] = texts[i].replace('】','')
    texts[i] = texts[i].replace('【','')
    texts[i] = texts[i].replace('。','')
    
    # words = pseg.cut(texts[i])      
    # word = [w for w,f in words]
    # 第二種不同的切法
    word =' '.join(jieba.cut(texts[i],cut_all=False)) 
    # print(texts[i])
    # print(word)
    return word
tok = keras.preprocessing.text.Tokenizer(10000)
max_length = 800
X2 = []
for i in range(0,10):

    listrow = cut_row(test_df,i)

    X2.append(listrow)
tok.fit_on_texts(X2)
arr2 = tok.texts_to_sequences(X2)
xtest = pad_sequences(arr2, maxlen=max_length, padding='post')

print(xtest)


