import pandas as pd
from flask import Flask,render_template,request
from sklearn.ensemble import RandomForestRegressor
import math
import pickle as pkl
df=pd.read_csv("apartment_cost_list.csv")
df.drop(columns=['Job #', 'House #', 'Curb Cut', 'Horizontal Enlrgmt', 'Vertical Enlrgmt'], inplace=True)
df['Year'] = df['Fully Permitted'].str[-4:]
df.drop(columns=['Fully Permitted'], inplace=True)
df['Initial Cost'] = df['Initial Cost'].str.replace('$', '').str.slice(stop=-3)
df['Initial Cost'] = df['Initial Cost'].astype(int)
train_data = df.dropna()
test_data = df[df['Year'].isnull()]
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(train_data.index.values.reshape(-1, 1), train_data['Year'])
predictions = regressor.predict(test_data.index.values.reshape(-1, 1))
predictions = [math.floor(p) for p in predictions]
df.loc[df['Year'].isnull(), 'Year'] = predictions
df['Year'] = df['Year'].astype(int)
df.drop(columns=['Street Name', 'Block', 'Bin #', 'Job Description', 'Zoning Dist1'], inplace=True)
df = df.drop_duplicates()
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
z=label_encoder.fit(df['Borough'].unique())
h=LabelEncoder()
y=h.fit(df['Job Type'].unique())
df=z.transform(df['Borough'])
df=y.transform(df['Job Type'])
def remove_outliers_iqr(df, columns):
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df
columns_to_check = ['Lot', 'Community - Board', 'Proposed Zoning Sqft', 'Enlargement SQ Footage']
df_clean = remove_outliers_iqr(df, columns_to_check)
df_clean.drop(columns=['Proposed Zoning Sqft', 'Enlargement SQ Footage'], inplace=True)
X = df_clean.drop('Initial Cost', axis=1)
y = df_clean['Initial Cost']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=0)
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
# Create a DecisionTreeRegressor object
dtree = DecisionTreeRegressor()

# Define the hyperparameters to tune and their values
param_grid = {
    'max_depth': [2, 4, 6, 8],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 3, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Create a GridSearchCV object
grid_search = GridSearchCV(dtree, param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the GridSearchCV object to the data
grid_search.fit(X_train, y_train)
from sklearn.tree import DecisionTreeRegressor
dtree = DecisionTreeRegressor(random_state=0, max_depth=2, max_features='log2', min_samples_leaf=1, min_samples_split=4)
dtree.fit(X_train, y_train)
pkl.dump(dtree,open('model.pkl','wb'))