import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split,KFold

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

mean_error=[]; std_error=[]
K_range = [2,3,4,5,6,7,8,9,10]
for K in K_range:
    model = KNeighborsRegressor(n_neighbors=K)
    temp=[]
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model.fit(X[train], Y[train])
        ypred = model.predict(X[test])
        from sklearn.metrics import mean_squared_error
        temp.append(mean_squared_error(Y[test],ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
import matplotlib.pyplot as plt

plt.errorbar(K_range,mean_error,yerr=std_error,linewidth=3)
plt.xlabel('K'); plt.ylabel('Mean square error')
plt.xlim((1,11))
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=50, test_size=0.3)

neigh = KNeighborsRegressor(n_neighbors=7)
neigh.fit(X_train, y_train)
y_tr=neigh.predict(X_train)
y_pred=neigh.predict(X_test)
y=neigh.predict(X)
print('train Mean Absolute Error:', mean_absolute_error(y_train, y_tr))
print('test Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('train Mean Squared Error:', mean_squared_error(y_train, y_tr))
print('test Mean Squared Error:', mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('kNN train score: ',neigh.score(X_train,y_train))
print('kNN test score: ',neigh.score(X_test,y_test))
plt.ylabel('Rating',fontsize=15)
plt.xlabel('Goals_pG',fontsize=15)
plt.title('kNN', fontsize=25)
plt.scatter(X_train[:,0],y_train,color='red')
plt.scatter(X_train[:,0],y_tr,color='blue')
plt.legend(["training data","prediction"])
plt.show()
plt.ylabel('Rating',fontsize=15)
plt.xlabel('Goals_pG',fontsize=15)
plt.title('kNN', fontsize=25)
plt.scatter(X_test[:,0],y_test,color='red')
plt.scatter(X_test[:,0],y_pred,color='blue')
plt.legend(["training data","prediction"])
plt.show()