import pandas as pd
import numpy as np  
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

data=pd.read_csv('cars_sampled.csv')
sns.set_style('darkgrid')
sns.set(rc={'figure.figsize':(11.7,8.27)})
data2=data.copy()
# print(data.info())


#sumarrizing the data
summary = data.describe(include='all')
pd.set_option('display.float_format', lambda x: '%.3f' % x)
#display maximum number of rows and columns
pd.set_option('display.max_columns', 100)
# print(summary)

#removing unwanted datas which are not in use
col=['name','dateCrawled','dateCreated','postalCode','lastSeen']
data=data.drop(columns=col,axis=1)


#removing duplicates
data.drop_duplicates(keep='first',inplace=True)
# print (data.duplicated().sum())

#removing null values
data.dropna(axis=0, inplace=True)
data.dropna(axis=1, inplace=True)
# print(data.isnull().sum())


# #variable yearOFRegistration
# print(data['yearOfRegistration'].value_counts().sort_index())
# print(sum(data['yearOfRegistration'] < 1950))
# print(sum(data['yearOfRegistration'] > 20168))

# #variable price
# print(data['price'].value_counts().sort_index())
# print(sum(data['price'] < 0))
# print(sum(data['price'] > 150000))

# #variable powerPS
# print(data['powerPS'].value_counts().sort_index())
# print(sum(data['powerPS'] < 0))
# print(sum(data['powerPS'] > 1000))

#-----------------------------------------------------------------------------------------------#   
#                                    working range of data                                      #  
# ----------------------------------------------------------------------------------------------#
data=data[
    (data.yearOfRegistration >= 1950) & 
    (data.yearOfRegistration <=2018) &
    (data.price >= 100) & (data.price <= 150000) &
    (data.powerPS >=10) &
    (data.powerPS <= 500)]
data['monthOfRegistration']/=12



# creating a new variable age bye subtractiong year of registraipn and adding month of registrartion
data['age']=(2018-data['yearOfRegistration']+data['monthOfRegistration'])
data['age']=round(data['age'],2)

#dropping year of registration and montofregistration

data=data.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)
# sns.distplot(data['age'])
# plt.show()

data_omit=data.dropna(axis=0, inplace=False)
data_omit=pd.get_dummies(data_omit, drop_first=True)
X1= data_omit.drop(columns=['price'],axis='columns',inplace=False)
Y1= data_omit['price']
prices=pd.DataFrame({"1. before" :Y1 , "2. after" :np.log(Y1)})
# print(prices)
prices.hist(bins=20)
# plt.show()
Y1=np.log(Y1)
X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, test_size=0.3, random_state=3)
# print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
base_pred= np.mean(Y_test)
# print("Base prediction: ", base_pred)
base_pred=np.repeat(base_pred, len(Y_test))
base_root_mean_error = np.sqrt(mean_squared_error(Y_test, base_pred))



# print("Base RMSE: ", base_root_mean_error)



Lgr=LinearRegression(fit_intercept=True)
model=Lgr.fit(X_train, Y_train)
data_pred=Lgr.predict(X_test)


#again calculatin root mean squared error against the y_test and data_pred


root_mean_error = np.sqrt(mean_squared_error(Y_test, data_pred))


# print("RMSE: ", root_mean_error)



#calculating r squared value

   
r_squared = model.score(X_test, Y_test)


# print("R-squared: ", r_squared)


#----------------------------------------------------------------------------#
#                 random forest with omitted data                            #
#----------------------------------------------------------------------------#


#model parameters
rf=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, max_features=None , min_samples_split=10 ,  min_samples_leaf=4)
model=rf.fit(X_train, Y_train)
data_pred=rf.predict(X_test)

#comuting root mean squared error
root_mean_error = np.sqrt(mean_squared_error(Y_test, data_pred))


# print("RMSE: ", root_mean_error)


#computing r squared value
r_squared_test = model.score(X_test, Y_test)
r_squared_train = model.score(X_train, Y_train)





# print("R-squared test: ", r_squared_test)
# print("R-squared train: ", r_squared_train)



#=============================================================================#
#                 model building from imputed data                            #
#=============================================================================#

data_imputed=data.apply(lambda x: x.fillna(x.median()) if x.dtype=='float' else x.fillna(x.value_counts().index[0]))


# print(data_imputed.isnull().sum())


data_imputed=pd.get_dummies(data_imputed, drop_first=True)
print(data_imputed)
x1= data_imputed.drop(columns=['price'],axis='columns',inplace=False)
y1= data_imputed['price']
y1=np.log(y1)
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(x1, y1, test_size=0.3, random_state=3)
# print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)       
base_pred= np.mean(Y_test1)
# print("Base prediction: ", base_pred)
base_pred=np.repeat(base_pred, len(Y_test1))
base_root_mean_error = np.sqrt(mean_squared_error(Y_test1, base_pred))
# print("Base RMSE: ", base_root_mean_error)  

Lgr=LinearRegression(fit_intercept=True)
model=Lgr.fit(X_train1, Y_train1)
data_pred=Lgr.predict(X_test1)
#again calculatin root mean squared error against the y_test and data_pred
root_mean_error = np.sqrt(mean_squared_error(Y_test1, data_pred))

print("RMSE: ", root_mean_error)

#calculating r squared value
r_squared = model.score(X_test1, Y_test1)


print("R-squared: ", r_squared)
