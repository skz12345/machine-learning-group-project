import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
df = pd.read_csv ("final.csv")
print (df.head( ))
X = df.drop(['Rating'],axis=1)
Y = df['Rating']
seed=50
X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size=0.3,
                                                    random_state=seed) 
rfr = RandomForestRegressor(n_estimators=20, # 20 trees
                            max_depth=4, # 5 levels
                            random_state=seed)

rfr.fit(X_train,y_train)
y_tr=rfr.predict(X_train)
y_pred=rfr.predict(X_test)
print('train Mean Absolute Error:', mean_absolute_error(y_train, y_tr))
print('test Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('train Mean Squared Error:', mean_squared_error(y_train, y_tr))
print('test Mean Squared Error:', mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('RF train score: ',rfr.score(X_train,y_train))
print('RF test score: ',rfr.score(X_test,y_test))

# Organizing feature names and importances in a DataFrame
features_df = pd.DataFrame({'features': rfr.feature_names_in_, 'importances': rfr.feature_importances_ })

# Sorting data from highest to lowest
features_df_sorted = features_df.sort_values(by='importances', ascending=False)

# Barplot of the result without borders and axis lines
g = sns.barplot(data = features_df_sorted, x='importances', y ='features', palette = "rocket")
sns.despine(bottom = True, left = True)
g.set_title('Feature importances')
g.set(xlabel = None)
g.set(ylabel = None)
g.set(xticks = [])
for value in g.containers:
    g.bar_label(value, padding=2)
plt.show()

# features = df.columns
# Obtain just the first tree
# first_tree = rfr.estimators_[0]

# plt.figure(figsize=(15,6))
# tree.plot_tree(first_tree,
#                feature_names=features,
#                fontsize=8, 
#                filled=True, 
#                rounded=True)
# plt.show()
Goals_pG=df.iloc[:,0]
Assists_pG=df.iloc[:,1]
Yel_pG=df.iloc[:,2]
Red_pG=df.iloc[:,3]
Shot_pG=df.iloc[:,4]
Pass_success_rate=df.iloc[:,5]
AerialsWon_pG=df.iloc[:,6]
X=np.column_stack((Goals_pG,Assists_pG,Yel_pG,Red_pG,Shot_pG,Pass_success_rate,AerialsWon_pG))
Y=df.iloc [:,7]
seed=50
X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                    test_size=0.3,
                                                    random_state=seed) 
rfr = RandomForestRegressor(n_estimators=20, # 20 trees
                            max_depth=4, # 4 levels
                            random_state=seed)

rfr.fit(X_train,y_train)
y_tr=rfr.predict(X_train)
y_pred=rfr.predict(X_test)
plt.ylabel('Rating',fontsize=15)
plt.xlabel('Goals_pG',fontsize=15)
plt.title('RF', fontsize=25)
plt.scatter(X_train[:,0],y_train,color='red')
plt.scatter(X_train[:,0],y_tr,color='blue')
plt.legend(["training data","prediction"])
plt.show()
plt.ylabel('Rating',fontsize=15)
plt.xlabel('Goals_pG',fontsize=15)
plt.title('RF', fontsize=25)
plt.scatter(X_test[:,0],y_test,color='red')
plt.scatter(X_test[:,0],y_pred,color='blue')
plt.legend(["training data","prediction"])
plt.show()
