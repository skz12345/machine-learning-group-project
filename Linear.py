import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

## Linear
print("LinearRegression")
linear=LinearRegression().fit(X,Y)
print(linear.intercept_,linear.coef_)
y_tr=linear.predict(X_train)
y_pred=linear.predict(X_test)
y=linear.predict(X)
print('train Mean Absolute Error:', mean_absolute_error(y_train, y_tr))
print('test Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('train Mean Squared Error:', mean_squared_error(y_train, y_tr))
print('test Mean Squared Error:', mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('linear train score: ',linear.score(X_train,y_train))
print('linear test score: ',linear.score(X_test,y_test))

plt.ylabel('Rating',fontsize=15)
plt.xlabel('Goals_pG',fontsize=15)
plt.title('Linear', fontsize=25)
plt.scatter(Goals_pG,Y,color='red')
plt.scatter(Goals_pG,y,color='blue')
plt.legend(["training data","prediction"])
plt.show()


