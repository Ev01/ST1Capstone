import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import make_scorer
# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle

import analysis
import handle_data

TARGET_VARIABLE = "log_price"
correlated_predictors = analysis.get_correlated_predictors()


def accuracy_score(orig, pred):
    """Calculate the accuracy of the predicted data based on the original. Intended to be used as a custom scoring function."""
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    #print('#'*70,'Accuracy:', 100-MAPE)
    return(100-MAPE)



def standardise_values(X):
    """Return a standardised version of the predictors."""
    # Both StandardScaler and MinMaxScaler give similar results
    #predictor_scaler = StandardScaler()
    predictor_scaler = MinMaxScaler()
    predictor_scaler_fit = predictor_scaler.fit(X)
    # Generate the standardized values of X
    X = predictor_scaler_fit.transform(X)
    return X



def get_processed_data(predictors=[]):
    """Read the dataframe and process it completely for use with machine learning."""
    df = handle_data.get_formatted_dataframe()
    handle_data.handle_all_outliers(df)
    handle_data.handle_null_values(df)
    df = handle_data.convert_columns_to_numeric(df)

    # If a list of predictors has not been assigned, default to the correlated predictors.
    if not predictors:
        predictors = correlated_predictors
    
    # Remove all columns that are not a selected predictor or target variable
    df = df[[*predictors, TARGET_VARIABLE]]

    X = df[predictors].values
    y = df[TARGET_VARIABLE].values

    X = standardise_values(X)

    return df, X, y


def evaluate_model_accuracy(reg_model, X, y, predictors, target_variable):
    """Train 70% of the data with the specified model, then print information on the accuracy of the model."""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    # Printing all the parameters of Linear regression
    print(reg_model)

    # Creating the model on Training Data
    LREG = reg_model.fit(X_train,y_train)
    prediction = LREG.predict(X_test)

    # Measuring Goodness of fit in Training data
    print('R2 Value:', metrics.r2_score(y_train, LREG.predict(X_train)))

    ###########################################################################
    print('\n##### Model Validation and Accuracy Calculations ##########')

    # Create a Dataframe to store the predictions
    predicted_column = 'predicted_'+target_variable
    testing_data_results = pd.DataFrame(data=X_test, columns=predictors)
    testing_data_results[target_variable] = y_test
    testing_data_results[predicted_column] = prediction

    # Print sample prediction values
    print(testing_data_results.head())

    # Calculate the error of each prediction, and store this in a new column
    prediction_difference = abs(testing_data_results[target_variable] - testing_data_results[predicted_column])
    testing_data_results['APE'] = 100 * prediction_difference / testing_data_results[target_variable]

    mean_APE = np.mean(testing_data_results['APE'])
    median_APE = np.median(testing_data_results['APE'])

    accuracy = 100 - mean_APE
    median_accuracy = 100 - median_APE
    print('Mean Accuracy on test data:', accuracy) # Can be negative sometimes due to outlier
    print('Median Accuracy on test data:', median_accuracy)

    # Custom Scoring accuracy calculation
    custom_scoring = make_scorer(accuracy_score, greater_is_better=True)

    # Running 10-Fold Cross validation on a given algorithm
    # Passing full data X and y because the K-fold will split the data and automatically choose train/test
    accuracy_values = cross_val_score(reg_model, X , y, cv=10, scoring=custom_scoring)
    print('\nAccuracy values for 10-fold Cross Validation:\n',accuracy_values)
    print('\nFinal Average Accuracy of the model:', round(accuracy_values.mean(),2))



def test_models():
    """Train the data using three different models: Linear regression, decision tree regressor, and random forest regressor."""
    # Note: Tree regressor is most accurate
    predictors = correlated_predictors
    df, X, y = get_processed_data(predictors=predictors)
    print("Testing linear regression model")
    reg_model = LinearRegression()
    evaluate_model_accuracy(reg_model, X, y, predictors, TARGET_VARIABLE)

    print("Testing Tree Regressor Model")
    reg_model = DecisionTreeRegressor(max_depth=5,criterion='friedman_mse')
    evaluate_model_accuracy(reg_model, X, y, predictors, TARGET_VARIABLE)

    print("Testing Random Forest Regressor")
    reg_model = RandomForestRegressor(max_depth=4, n_estimators=400,criterion='friedman_mse')
    evaluate_model_accuracy(reg_model, X, y, predictors, TARGET_VARIABLE)


def test_final_model():
    """Test the final model using only the most important predictors and print its results."""
    # The most important predictors based on the results of plot_feature_importance()
    important_predictors = ["bedrooms", "accommodates", "bathrooms"]
    # The tree regressor was the most accurate out of the ones tested
    reg_model = DecisionTreeRegressor(max_depth=5,criterion='friedman_mse')
    #target_variable = "log_price"

    df, X, y = get_processed_data(predictors=important_predictors)
    evaluate_model_accuracy(reg_model, X, y, important_predictors, TARGET_VARIABLE)



def train_final_model():
    """Return a model that has been trained using all of the data with only the most important predictors."""
    important_predictors = ["bedrooms", "accommodates", "bathrooms"]
    # The tree regressor was the most accurate out of the ones tested
    reg_model = DecisionTreeRegressor(max_depth=5,criterion='friedman_mse')
    #target_variable = "log_price"

    df, X, y = get_processed_data(important_predictors)

    # Train model using all data
    final_model = reg_model.fit(X, y)
    return final_model


def save_model_to_file(model):
    """Save the model to a picle file for later use."""
    with open('final_model.pkl', 'wb') as file:
        pickle.dump(model, file)


def predict_result(input_data):
    """Return a dataframe with multiple predictions of log_price based on the input data."""
    predictors = ["bedrooms", "accommodates", "bathrooms"]
    X = input_data[predictors]
    X = standardise_values(X)

    with open('final_model.pkl', 'rb') as file:
        prediction_model = pickle.load(file)
    
    # Genrating Predictions
    prediction = prediction_model.predict(X)
    prediction_result = pd.DataFrame(prediction, columns=['Prediction'])
    return prediction_result


def generate_prediction(bedrooms, accommodates, bathrooms):
    """Return a JSON string with a single prediction."""
    input_data = pd.DataFrame(data=[[bedrooms, accommodates, bathrooms]], columns=["bedrooms", "accommodates", "bathrooms"])
    predictions = predict_result(input_data)
    # Return the first (and only) prediction
    d = {"Prediction": float(predictions.values[0])}
    return json.dumps(d)


def test_sample_data():
    """Print the predictions of some made up sample data."""
    sample_data = pd.DataFrame(data = [[3, 4, 2], [1, 1, 1], [4, 6, 2.5]], columns=["bedrooms", "accommodates", "bathrooms"])
    print(predict_result(sample_data))


if __name__ == "__main__":
    test_models()
    #plot_feature_importance()
    #test_final_model()
    #final_model = train_final_model()
    #save_model_to_file(final_model)
    #test_sample_data()
    #generate_prediction(3, 2, 4)