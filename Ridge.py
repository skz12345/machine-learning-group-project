import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures

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

mean_error=[]; std_error=[]
polynomial=[1,2]
for i in polynomial:
    po=PolynomialFeatures(degree=i)
    features_power=po.fit_transform(X)
    Xtrain=features_power
    Ytrain=Y
    model=Ridge(alpha=2,max_iter=10000)
    model.fit(Xtrain,Ytrain)
    temp=[]
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model.fit(Xtrain[train], Ytrain[train])
        ypred = model.predict(Xtrain[test])
        from sklearn.metrics import mean_squared_error
        temp.append(mean_squared_error(Ytrain[test],ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())
import matplotlib.pyplot as plt
plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
plt.errorbar(polynomial,mean_error,yerr=std_error,linewidth=3)
plt.xlabel('pow'); plt.ylabel('Mean square error')
plt.xlim((0,3))
plt.show()

## Ridge

mean_error=[];std_error=[]
Ci_range=[0.01,0.1,0.5,1,2,3,4,5]
for Ci in Ci_range:
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1/(2*Ci))
    temp=[]
    kf=KFold(n_splits=5)
    for train,test in kf.split(X):
        model.fit(X[train],Y[train])
        ypred=model.predict(X[test])
        temp.append(mean_squared_error(Y[test],ypred))
    mean_error.append(np.array(temp).mean())
    std_error.append(np.array(temp).std())

print("temp is",temp)
print("std_error is ",std_error)

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
plt.errorbar(Ci_range,mean_error,yerr=std_error,linewidth=3)
plt.title('Ridge', fontsize=25)
plt.xlabel('C')
plt.ylabel('Mean square error')
plt.show()
# alpha_range=[50,5,1,0.5,0.25,0.167,0.125,0.1]
# ridgeCV= RidgeCV(alphas=alpha_range,cv=5).fit(X,Y)
# best_alpha = ridgeCV.alpha_
# print("For Ridge, best_alpha is",best_alpha)
# print("best C is",1/(2*best_alpha))

ridge=Ridge(alpha=0.25)
po=PolynomialFeatures(degree=2)
X_train_2=po.fit_transform(X_train)
X_test_2=po.fit_transform(X_test)
X_2=po.fit_transform(X)
ridge.fit(X_train_2,y_train)
y_tr=ridge.predict(X_train_2)
y_pred=ridge.predict(X_test_2)
y=ridge.predict(X_2)
print('Ridge coefficients: ',ridge.coef_)
print('train Mean Absolute Error:', mean_absolute_error(y_train, y_tr))
print('test Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('train Mean Squared Error:', mean_squared_error(y_train, y_tr))
print('test Mean Squared Error:', mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Ridge train score: ',ridge.score(X_train_2,y_train))
print('Ridge test score: ',ridge.score(X_test_2,y_test))
plt.ylabel('Rating',fontsize=15)
plt.xlabel('Goals_pG',fontsize=15)
plt.title('Ridge', fontsize=25)
plt.scatter(X_train[:,0],y_train,color='red')
plt.scatter(X_train[:,0],y_tr,color='blue')
plt.legend(["training data","prediction"])
plt.show()
plt.ylabel('Rating',fontsize=15)
plt.xlabel('Goals_pG',fontsize=15)
plt.title('Ridge', fontsize=25)
plt.scatter(X_test[:,0],y_test,color='red')
plt.scatter(X_test[:,0],y_pred,color='blue')
plt.legend(["training data","prediction"])
plt.show()