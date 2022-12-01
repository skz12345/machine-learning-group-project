import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv ("final.csv")
print (df.head( ))
Goals_pG=df.iloc[:,0]
Assists_pG=df.iloc[:,1]
Yel_pG=df.iloc[:,2]
Red_pG=df.iloc[:,3]
Shot_pG=df.iloc[:,4]
Pass_success_rate=df.iloc[:,5]
AerialsWon_pG=df.iloc[:,6]
X=np.column_stack((Goals_pG,Assists_pG,Yel_pG,Red_pG,Shot_pG,Pass_success_rate,AerialsWon_pG))
Y=df.iloc [:,7]

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=50, test_size=0.3)

dummy=DummyRegressor(strategy="mean").fit(X_train,y_train)
y_tr=dummy.predict(X_train)
y_pred=dummy.predict(X_test)
print('train Mean Absolute Error:', mean_absolute_error(y_train, y_tr))
print('test Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('train Mean Squared Error:', mean_squared_error(y_train, y_tr))
print('test Mean Squared Error:', mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('kNN train score: ',dummy.score(X_train,y_train))
print('kNN test score: ',dummy.score(X_test,y_test))

plt.ylabel('Rating',fontsize=15)
plt.xlabel('Goals_pG',fontsize=15)
plt.title('Dummy', fontsize=25)
plt.scatter(X_train[:,0],y_train,color='red')
plt.scatter(X_train[:,0],y_tr,color='blue')
plt.legend(["training data","prediction"])
plt.show()
plt.ylabel('Rating',fontsize=15)
plt.xlabel('Goals_pG',fontsize=15)
plt.title('Dummy', fontsize=25)
plt.scatter(X_test[:,0],y_test,color='red')
plt.scatter(X_test[:,0],y_pred,color='blue')
plt.legend(["training data","prediction"])
plt.show()