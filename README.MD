# Airbnb Price Predictor

## How to run

- run `train.py` to test all the models and then train and save the final model to a file. The program will test three different regression models: the linear regression model, tree regressor model, and random forest regressor model. Note that the random forest regressor model takes relatively long to test. The program will then cut down the features to the most useful ones and do a final test with these (see `plot_feature_importance()` in `graphs.py`). Finally, the program will train the model using all of the data and save this to `final_model.pkl`.

- run `graphs.py` to plot every graph that I have used in the main document. The next graph will only be shown after the window of the currently shown graph is closed.

- run `flask_api.py` to start a web api that can predict new data. Copy and paste this url into a web browser to test it with some sample data: http://127.0.0.1:9000/prediction_api?bedrooms=3&accommodates=4&bathrooms=2

## Contents of other files

- `analysis.py` contains functions related to correlation analysis of continous and categorical predictors.
- `handle_data.py` contains functions related to processing and preparing the data for use with machine learning. It handles tasks such as removing missing values and outliers, and converting categorical/qualitative variables to numeric ones.