import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os
import seaborn as sns

df=pd.read_csv('Breast_Cancer.csv')
df.info()

df.isnull().sum()

df.head(10)


df.drop('id',axis=1,inplace=True)
df.drop('Unnamed: 32',axis=1,inplace=True)


df.diagnosis.unique()

df['diagnosis']= df['diagnosis'].map({'M':1,'B':0})

plt.hist(df['diagnosis'])
plt.title('Diagnosis (M=1 , B=0)')
plt.show()


features_mean=list(df.columns[1:11])

dfM= df[df['diagnosis']==1]
dfB=df[df['diagnosis']==0]
