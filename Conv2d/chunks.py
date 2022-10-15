import pandas as pd
from sklearn.model_selection import train_test_split


data=pd.read_csv('./Conv2d/boston.csv')
batch_size=20

X=data.drop('price',axis=1)
Y=data['price']

def chunker(seq,size):
        for p in range(0,len(seq),size):
             yield seq.iloc[p:p+size]

# for i in chunker(data,20):
#     print(i)

print(data[10:20])