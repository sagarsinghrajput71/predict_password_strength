import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv('D:\python\predicting password strenght/data.csv',error_bad_lines=False)
data.head()
data['strength'].unique()
data.isnull().sum()
data[data['password'].isnull()]
data.dropna(inplace=True)

data.isnull().sum()
sns.countplot(data['strength'])
password_tuple=np.array(data)
password_tuple
import random
random.shuffle(password_tuple)
x=[lables[0]for lables in password_tuple]
y=[lables[1]for lables in password_tuple]
x


def word_divide_char(inputs):
    character = []
    for i in inputs:
        character.append(i)
    return character
word_divide_char('kzde5577')

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(tokenizer=word_divide_char)
X=vectorizer.fit_transform(x)
X.shape
vectorizer.get_feature_names()
first_document_vector=X[0]
first_document_vector
first_document_vector.T.todense()
df=pd.DataFrame(first_document_vector.T.todense(),index=vectorizer.get_feature_names(),columns=['TF-IDF'])
df.sort_values(by=['TF-IDF'],ascending=False)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train ,y_test=train_test_split(X,y,test_size=0.2)
X_train.shape
from sklearn.linear_model import LogisticRegression

clf=LogisticRegression(random_state=0,multi_class='multinomial')
clf.fit(X_train,y_train)
dt=np.array(['sagar12'])
pred=vectorizer.transform(dt)
clf.predict(pred)
y_pred=clf.predict(X_test)
y_pred
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))