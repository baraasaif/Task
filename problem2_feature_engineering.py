import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#  بيانات وهمية للتجربة
data = pd.DataFrame({
    'age': [25, 30, np.nan, 40, 35],
    'income': [50000, 60000, 55000, np.nan, 65000],
    'gender': ['male', 'female', 'female', 'male', np.nan],
    'target': [1, 0, 1, 0, 1]
})


X = data.drop('target', axis=1)
y = data['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


numeric_features = ['age', 'income']
categorical_features = ['gender']


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])


model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('regressor', LinearRegression())
])


model.fit(X_train, y_train)


predictions = model.predict(X_test)


print("Predictions:", predictions)
print("Processed feature shape:", model.named_steps['preprocessing'].transform(X_test).shape)