import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

data = pd.read_csv('American_Housing_Data_20231209.csv')
features = ['County', 'Living Space', 'Beds', 'Baths']

X = data.drop('Price', axis=1)
y = data['Price']
X_selected = X[features]

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

numeric_features = ['Living Space', 'Beds', 'Baths']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['County']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'regressor__n_estimators': [50, 100, 150],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4],
    'regressor__max_features': ['sqrt', 'log2']    
}

# Create a pipeline for the random forest model
random_forest_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('regressor', RandomForestRegressor(random_state=42))])

# Create a RandomizedSearchCV object
random_search = RandomizedSearchCV(random_forest_pipeline, param_distributions=param_grid, n_iter=10, cv=5, random_state=42)

# Fit the RandomizedSearchCV object to the training data
random_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = random_search.best_params_

# Print the best hyperparameters
print("Best Hyperparameters:", best_params)

# Get the best model
best_model = random_search.best_estimator_

# Print the best model
print("Best Model:", best_model)

# Make predictions on the test set
y_pred_random_forest = best_model.predict(X_test)

# Evaluate the random forest model
mse_random_forest = mean_squared_error(y_test, y_pred_random_forest)
mae_random_forest = mean_absolute_error(y_test, y_pred_random_forest)    
r2_random_forest = r2_score(y_test, y_pred_random_forest)

print("Random Forest Model Metrics:")
print("Mean Squared Error:", mse_random_forest)
print("Mean Absolute Error:", mae_random_forest)
print("R-squared:", r2_random_forest)
