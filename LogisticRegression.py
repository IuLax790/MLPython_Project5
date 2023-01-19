import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import warnings
def warn(*args,**kwargs):
    pass
warnings.warn = warn
Heart_Attack = pd.read_csv("C:\\Information_Science\\My_projects\\heart.csv")

columns_target = ['output']
columns_train = ['age','cp','chol','oldpeak','slp', \
                 'caa'                 ]
X = Heart_Attack[columns_train]
y = Heart_Attack[columns_target]
print(X)
print(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

clf = LogisticRegression()
clf.fit(X_train,y_train)
preds = clf.predict(X_test)
score = clf.score(X_test,y_test)
cm = metrics.confusion_matrix(y_test,preds)
print(cm)

plt.figure(figsize=(9,9))
print(sns.heatmap(cm,annot=True,fmt=".3f",linewidth=.5,\
            square=True,cmap='viridis_r'))
plt.xlabel('Actual Table')
plt.ylabel('Predicted Table')
all_sample_title = 'Accuracy Score: {0}'.format(score)
print(plt.title(all_sample_title,size=15))

index = 0
misclassifiedIndex = []
for predict, actual in zip(preds,y_test):
    if predict==actual:
        misclassifiedIndex.append(index)
    index+=1
print(plt.figure(figsize=(20,1)))
for plotIndex,wrong in enumerate(misclassifiedIndex[0:4]):
    print(plt.subplot(1,4,plotIndex+1))
    print(plt.imshow(np.reshape(X_test[wrong],(8,8)),cmap=plt.cm.gray))
    print(plt.title("Predicted: {},Actual:{}".format(preds[wrong],y_test[wrong]),fontsize=20))


sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
px.histogram(Heart_Attack, x='chol', title='cholesterol vs. Heart Attack', color='RainToday')
