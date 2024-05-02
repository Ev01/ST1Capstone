import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import make_scorer
# Importing cross validation function from sklearn
from sklearn.model_selection import cross_val_score


import analysis
import handle_data


def categorical_to_numeric(df, column, values_list):
    """Converts each value in the values list to its index in the list, then applies this to the column."""
    for i, value in enumerate(values_list):
        df.loc[df[column] == value, column] = i


def ml_preprocess(df):
    #df = handle_data.get_formatted_dataframe()

    # Convert all values of super_strict_30 or super_strict_60 in cancellation_policy to strict.
    df.loc[(df["cancellation_policy"] == "super_strict_30") | (df["cancellation_policy"] == "super_strict_60"), "cancellation_policy"] = "strict"

    predictors = analysis.get_correlated_predictors()
    df = df[[*predictors, "log_price"]]
    #print(list(df["cleaning_fee"].unique()))
    categorical_to_numeric(df, "cancellation_policy", ["flexible", "moderate", "strict"])
    categorical_to_numeric(df, "host_identity_verified", ["f" , "t"])
    categorical_to_numeric(df, "instant_bookable", ["f", "t"])
    # Convert cleaning fee from boolean to numeric
    df["cleaning_fee"] = df["cleaning_fee"].astype(int)

    return df, predictors



# Defining a custom function to calculate accuracy
# Make sure there are no zeros in the Target variable if you are using MAPE
def accuracy_score(orig, pred):
    MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
    #print('#'*70,'Accuracy:', 100-MAPE)
    return(100-MAPE)


#def get_processed_data():

#def ev


def ml():
    df = handle_data.get_formatted_dataframe()
    handle_data.handle_all_outliers(df)
    handle_data.handle_null_values(df)
    df, predictors = ml_preprocess(df)
    target_variable = "log_price"

    X = df[predictors].values
    y = df[target_variable].values

    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=428)

    ### Sandardization of data ###

    # Choose either standardization or Normalization
    # Choose between standardization and MinMAx normalization
    #predictor_scaler = StandardScaler()
    predictor_scaler = MinMaxScaler()
    # Storing the fit object for later reference
    predictor_scaler_fit = predictor_scaler.fit(X)
    # Generating the standardized values of X
    X = predictor_scaler_fit.transform(X)
    # Split the data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    # Sanity check for the sampled data
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)


    #Multiple Linear Regression
    reg_model = LinearRegression()

    # Printing all the parameters of Linear regression
    print(reg_model)

    # Creating the model on Training Data
    LREG = reg_model.fit(X_train,y_train)
    prediction = LREG.predict(X_test)

    # Measuring Goodness of fit in Training data
    print('R2 Value:', metrics.r2_score(y_train, LREG.predict(X_train)))

    ###########################################################################
    print('\n##### Model Validation and Accuracy Calculations ##########')

    # Printing some sample values of prediction
    predicted_column = 'predicted_'+target_variable
    testing_data_results = pd.DataFrame(data=X_test, columns=predictors)
    testing_data_results[target_variable] = y_test
    testing_data_results[predicted_column] = prediction

    # Printing sample prediction values
    print(testing_data_results.head())

    # Calculating the error for each row
    prediction_difference = abs(testing_data_results[target_variable] - testing_data_results[predicted_column])
    testing_data_results['APE'] = 100 * prediction_difference / testing_data_results[target_variable]

    mean_APE = np.mean(testing_data_results['APE'])
    median_APE  =np.median(testing_data_results['APE'])

    accuracy = 100 - mean_APE
    median_accuracy= 100 - median_APE
    print('Mean Accuracy on test data:', accuracy) # Can be negative sometimes due to outlier
    print('Median Accuracy on test data:', median_accuracy)

    # Custom Scoring MAPE calculation
    custom_scoring = make_scorer(accuracy_score, greater_is_better=True)

    # Running 10-Fold Cross validation on a given algorithm
    # Passing full data X and y because the K-fold will split the data and automatically choose train/test
    accuracy_values = cross_val_score(reg_model, X , y, cv=10, scoring=custom_scoring)
    print('\nAccuracy values for 10-fold Cross Validation:\n',accuracy_values)
    print('\nFinal Average Accuracy of the model:', round(accuracy_values.mean(),2))
    





if __name__ == "__main__":
    ml()