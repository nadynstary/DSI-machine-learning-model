#full code
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor , GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 

#PART A Q1: data integrity check 
us=pd.read_csv('DSI_Dataset.csv')

print('shape:',us.shape)
print('info:',us.info())

print(us.head())
print(us.tail())
print(us.null())
print(       )
print(us.describe())
print(us.duplicated().sum())


#b                             ======================= Question 2=========================
#PART A Q2: scatter and histomap
plt.figure(figsize=(7, 5))
sns.scatterplot(x='DSI_target_0_100', y='CO2_emission_kilotons', data=us)
plt.xlabel=('DSI_target_0_100') 
plt.ylabel=('CO2 emission kilotons')       
plt.title=('CO2 emission kilotons vs DSI target')
plt.show()

plt.figure(figsize=(7, 5))
sns.histplot(us['Green_area_per_capita_m2'], kde=True, bins=20)
plt.title=('Green area per capita m2')
plt.xlabel=('Green areaa per capita m2')     
plt.ylabel=('Frequency')               
plt.show()


#PART A Q2: correlation heatmap and pairplot 
numeric_us=us.select_dtypes(include=['int64','float64'])

plt.figure(figsize=(12,8))
sns.heatmap(numeric_us.corr(), annot=True, cmap="PiYG", fmt='.2f')
plt.title=('Correlation Heatmap')
plt.show()

sns.pairplot(numeric_us)
plt.suptitle('Pair Plot of Dataset Features', y=1.02)
plt.show()
#                    ===================================== Question 3=======================================
#PART B Q3: filling the missing values & data cleansing (mode and median fill)
us=us.dropna(thresh=us.shape[1]*0.50)
for col in us.columns:
    if us[col].dtype=='object':
        us[col]=us[col].fillna(us[col].mode()[0])
    else:
        us[col]=us[col].fillna(us[col].median())
        
#                    ===================================== Question 4 =======================================
#PART B Q4: two derived features:
us['CO2_per_traffic']=us['Traffic_index_0_100']/ us['CO2_emission_kilotons']
us['recycling_per_denisty']=us['Waste_recycling_rate_pct']/us['Population_density_people_per_km2']*10
us.head()


#                    ===================================== Question 5 =======================================
#PART B Q5: feature scaling
#(splitting the data into two halves)

us = us.dropna(thresh=us.shape[1] * 0.50)

# Shuffle before splitting
us = us.sample(frac=1, random_state=42).reset_index(drop=True)


half = len(us) // 2
us_scaling = us.iloc[:half].copy()      
us_predict = us.iloc[half:].copy()      


# -----------  PART 1: Predictive Imputation -------------


# Select categorical and numerical columns
cat_cols=us_predict.select_dtypes(include='object').columns
num_cols=us_predict.select_dtypes(exclude='object').columns

for col in cat_cols:
    us_predict[col] = us_predict[col].fillna(us_predict[col].mode()[0])


us_encoded = pd.get_dummies(us_predict, drop_first=True)

# Predictive imputation for missing numerical columns
for col in num_cols:
    if us_predict[col].isna().sum() > 0:
        print(f"Imputing missing values for: {col}")

        # rows with and without missing values
        train_rows = us_predict[col].notna()
        pred_rows = us_predict[col].isna()

        # training data
        X_train = us_encoded[train_rows].drop(columns=[col])
        y_train = us_predict.loc[train_rows, col]

        # data to predict
        X_pred = us_encoded[pred_rows].drop(columns=[col])

        # model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # predict
        predicted_vals = model.predict(X_pred)

        # fill values
        us_predict.loc[pred_rows, col] = predicted_vals

# -----------  PART 2: MinMax + Standard Scaling ----------


numeric_scaling = us_scaling.select_dtypes(include=['float64', 'int64'])

# MinMax scaling
minmax_scaler = MinMaxScaler()
us_scaling_minmax = pd.DataFrame(
    minmax_scaler.fit_transform(numeric_scaling),
    columns=numeric_scaling.columns
)

# Standard scaling
standard_scaler = StandardScaler()
us_scaling_standard = pd.DataFrame(
    standard_scaler.fit_transform(numeric_scaling),
    columns=numeric_scaling.columns
)

# save scaling output back into us_scaling
us_scaling.loc[:, numeric_scaling.columns] = us_scaling_standard


# -------------------- merge back-------------------------


us_final = pd.concat([us_scaling, us_predict], axis=0).reset_index(drop=True)

print("\nFinal dataset shape =", us_final.shape)
print(us_final.head())



#PCA:
pca=PCA(n_components=2)
pca_result=pca.fit_transform(stan)
pca_us=pd.DataFrame({'PCA_1':pca_result[:,0],'PCA_2':pca_result[:,1]})

#visualization:
plt.figure(figsize=(7,5))
plt.scatter(pca_us['PCA_1'],pca_us['PCA_2'],color='purple',marker='s')
plt.xlabel=('PCA1')
plt.ylabel=('PCA2')
plt.title=('PCA visualization')
print('PCA visulization')
plt.show()
#              ===================================== Question 6, 7, 8 =======================================
#PART C Q6 & Q7 & Q8: Model Training / Ensemble /Evalution :
#model training 
target = 'DSI_target_0_100'
X=us.drop(columns=[target])
y=us[target]
X=X.select_dtypes(include=['float64','int64'])


#data splitting for training and testing
X_train,X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#modeling
models={'Linear Regression:':LinearRegression(),'Decision Tree:':DecisionTreeRegressor(),'SVM:':SVR(),'Random Forest:':RandomForestRegressor(),
        'Gradient Boosting:':GradientBoostingRegressor()}
results={}

for name,model in models.items():
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    mean_abs_value = mean_absolute_error(y_test,predictions)
    RMSE = np.sqrt(mean_squared_error(y_test,predictions))
    R2 = r2_score(y_test, predictions)
    
    results[name]=[mean_abs_value,RMSE,R2]




    
results_us=pd.DataFrame(results,index=['MAE','RMSE','R²']).T
print('the table:','\n', results_us)

