import jieba
import pandas as pd
import numpy as np
# from IPython.display import display
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
LABELS = ['幼兒園(0歲~5歲)', '小學(6歲~11歲)', '中學(12歲~14歲)', '高中(15歲~18歲)', '大學(18歲~22歲)', '壯年(24-39歲)', '中年(40-64歲)',
         '老年(65歲以上)']


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

def convert_testing_df_label(df): #把原本多類別標記的資料做multi-hot encoding
    del_list = []
    df['Label1'] = df['Label1'].str.replace('\n','')
    print(df['Label1'])
    for label in LABELS:
      df[label] = 0

    for i in range(df.shape[0]):
        for k in ['Label1', 'Label2']:
            if (type(df[k][i]) == str):
                temp = df[k][i].split(', ')
                if (temp == ['適合全年齡層']):
                    temp = LABELS
                for j in temp:
                    df[j][i] = df[j][i] + 1
            else:
                del_list.append(i)
    print(df['Label1'])
    # delete unlabeled data
    df = df.drop(del_list, axis=0) 
    for i in LABELS:
      df[i][df[i] < 2] = 0
      df[i][df[i] > 0] = 1  
    return df 

train_dataset_path = 'TestingData.xlsx'
train_df = load_df(train_dataset_path)

test_dataset_path = 'TrainingData.xlsx'
test_df = load_df(test_dataset_path)
test_df = convert_testing_df_label(test_df)
def load_tokenizer(df):
    vec = TfidfVectorizer()
    texts = []
    for index, row in df.iterrows():
        text = row['name'] + ' ' + row['description'] + ' ' + row['category'] + ' ' + row['owner name'] + ' ' + row[
            'location']
        cut_result = ' '.join(jieba.cut(text, cut_all=False, HMM=True))
        texts.append(cut_result)
    return vec.fit_transform(texts)

def load_dataset(df, vec, y_key): #把不同feature合併成一個字串
    x_list = []
    y_list = []

    for index, row in df.iterrows():
        text = row['name'] + ' ' + row['description'] + ' ' + row['category'] + ' ' + row['owner name'] + ' ' + row[
            'location']
        cut_result = ' '.join(jieba.cut(text, cut_all=False, HMM=True))
        x_list.append(cut_result)
        y_list.append(row[y_key])
    
    return x_list, np.array(y_list)


tokenizer = load_tokenizer(train_df)
vec = TfidfVectorizer()

X_train_, y_train = load_dataset(train_df, vec, 'Label')
X_train = vec.fit_transform(X_train_)
X_test_, y_test = load_dataset(test_df, vec, '高中(15歲~18歲)')
X_test = vec.transform(X_test_)

def evaluation(y_test, y_predict_scores, threshold):
    y_test = np.asarray(y_test)
    y_pred = np.select([y_predict_scores < threshold, y_predict_scores >= threshold],
                       [np.zeros_like(y_predict_scores), np.ones_like(y_predict_scores)])

    accuracy = accuracy_score(y_test, y_pred)  # 沒有average參數
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1score = f1_score(y_test, y_pred, average='binary')

    return accuracy, precision, recall, f1score
    
model = MultinomialNB()
model.fit(X_train, y_train)
test_predict_scores = model.predict(X_test)
accuracy, precision, recall, f1score = evaluation(y_test, test_predict_scores, 0.5)

print('accuracy :', accuracy)
print('precision:', precision)
print('recall   :', recall)
print('f1 score :', f1score)