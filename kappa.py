import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score
df = pd.read_csv('BinaryClassificationTrainingData_Team6 - training.csv') 
#kappa function 
#輸入: 必須為2個相同長度的二維串列(第一個維度同長)還有要搜尋的key值
#輸出: kappa值
def kappa(list1,list2,key):
    m=[]
    m2=[]
    #把label1值做搜尋key，再append到m裡
    for i in range(0,100):
        size = len(list1[i])
        b = 0
        #在內部串列搜尋
        for j in range(0,size):
            if(list1[i][j] == key):
                b = 1
        #防止重複寫入m        
        if(b==1):
            m.append(1)
            b = 0
        else:
            m.append(0)
    #把label2值做搜尋key，再append到m2裡        
    for i in range(0,100):
        size = len(list2[i])
        b = 0
        #在內部串列搜尋
        for j in range(0,size):
            if(list2[i][j] == key):
                b = 1
        #防止重複寫入m2     
        if(b==1):
            m2.append(1)
            b = 0
        else:
            m2.append(0)
    return cohen_kappa_score(m, m2)
#資料處理 Label1
df['Label1'] = df['Label1'].str.replace('\r','') 
df['Label1'] = df['Label1'].str.replace('\n','')
df['Label1'] = df['Label1'].str.replace(' ','')
df['Label1'] = df['Label1'].str.split(",")
#資料處理 Label2
df['Label2'] = df['Label2'].str.replace('\r','')
df['Label2'] = df['Label2'].str.replace('\n','')
df['Label2'] = df['Label2'].str.replace(' ','')
df['Label2'] = df['Label2'].str.split(",")
#設定要搜尋的key
key = "適合全年齡層"
ans = kappa(df['Label1'], df['Label2'],key)
print(ans)

