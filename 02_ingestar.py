quakes=pd.read_csv("quakes.csv")
X=quakes[['lat','long','depth','stations']]
y=quakes['mag']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
X_train_2,X_validate,y_train_2,y_validate=train_test_split(X_train,y_train,test_size=0.1)
df_train= X_train_2[['lat','long','depth','stations']].copy()
df_train['mag'] = y_train_2.copy()
df_train.to_csv("train.csv",index=False,columns=['lat','long','depth','stations','mag'])
df_validate= X_validate[['lat','long','depth','stations']].copy()
df_validate['mag'] = y_validate.copy()
df_validate.to_csv("validate.csv",index=False,columns=['lat','long','depth','stations','mag'])
df_test= X_test[['lat','long','depth','stations']].copy()
df_test['mag'] = y_test
df_test.to_csv("test.csv",index=False,columns=['lat','long','depth','stations','mag'])
