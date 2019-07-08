
linear regression: here output is regression
it is machine learning algorithum where its output depends on single indepedent input paratemer
====================================================================================================
multiple regression : output of it depends on  more than 1 independent inputs parameters

=======================================================================================================

import pandas as pd
from sklearn.linear_model import LinearRegression

data=pd.read_csv('/home/cbit/Desktop/gpa.csv')
print(data.head())


X=data.iloc[:,:-1]
y=data.iloc[:,1]


from sklearn.model_selection  import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)

 

model = LinearRegression()
model.fit(X_train, y_train)
yy=model.predict(X_test)
print(yy)

#accuracy
error=0
ss=list(y_test)
j=0
for i in yy:
    error=error+abs(i-ss[j])
    j=j+1
print(error)
    
  






























































