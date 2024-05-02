import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np


def get_formatted_dataframe():
    """Read the csv and cleans it up so that it can be graphed/analysed."""
    df = pd.read_csv("data.csv")
    # Remove any duplicate rows in the dataset
    df.drop_duplicates(inplace=True)

    # Convert host_response_rate from a percentage inside a string to a float between 0 and 1.
    df["host_response_rate"] = df["host_response_rate"].str.rstrip("%").astype('float') / 100.0

    # Convert date date columns to pandas datetime objects.
    df["host_since"] = pd.to_datetime(df["host_since"])
    df["last_review"] = pd.to_datetime(df["last_review"])
    df["first_review"] = pd.to_datetime(df["first_review"])

    # Drop all rows where the log_price is 0 (in this dataset there is only one)
    df = df.drop(list(df.loc[df["log_price"] == 0].index))

    return df


def print_null_percentage(df):
    """Print the percentage of null values for each column in the dataframe."""
    # isnull converts values to 1 if they are null, or 0 if not null.
    print(df.isnull().mean() * 100)



def get_outlier_upper_limit_sd(series):
    # Any value above this number is considered an outlier
    return series.mean() + series.std() * 3

def get_outlier_lower_limit_sd(series):
    # Any value below this number is considered an outlier
    return series.mean() - series.std() * 3


def is_outlier_sd(series):
    """Return a series with each value replaced with True if it is an outlier, or False if it isn't."""
    # Anything outside outside of three standard deviations is an outlier.
    upper_limit = get_outlier_upper_limit_sd(series)
    lower_limit = get_outlier_lower_limit_sd(series)

    outliers = (series >= upper_limit) | (series <= lower_limit)
    
    return outliers



def get_outliers_sd(series):
    """Return the rows in the series that are outliers."""
    return series.loc[is_outlier_sd(series)]



def outliers_to_limit(df, column, convert_to_int=True):
    """
    Use the winsorising method to convert all outliers in the column to the upper/lower limit.
    
    Setting convert_to_int to True will convert the upper and lower limits to ints. This should be 
    used when the column is categorical.
    """
    # Anything above the upper limit or below the lower limit is considered an outlier
    upper_limit = get_outlier_upper_limit_sd(df[column])
    lower_limit = get_outlier_lower_limit_sd(df[column])
    if convert_to_int:
        upper_limit = int(upper_limit)
        lower_limit = int(lower_limit)

    # Convert any value that is an outlier above the mean to the upper limit
    df.loc[is_outlier_sd(df[column]) & (df[column] > df[column].mean()), column] = upper_limit
    # Convert any value that is an outlier below the mean to the lower limit
    df.loc[is_outlier_sd(df[column]) & (df[column] < df[column].mean()), column] = lower_limit



def handle_all_outliers(df):
    """Handle the outliers for all columns that have outliers."""
    outliers_to_limit(df, "host_response_rate")
    outliers_to_limit(df, "review_scores_rating")
    outliers_to_limit(df, "accommodates")
    outliers_to_limit(df, "beds")
    outliers_to_limit(df, "bedrooms")
    outliers_to_limit(df, "bathrooms")


def fill_na_with_mode(df, column):
    """Replace all missing values in the column with the mode. This should be used for categorical variables."""
    mode = df[column].mode()[0]
    df.fillna({column: mode}, inplace=True)

def fill_na_with_median(df, column):
    """Replace all missing values in the column with the median. This should be used for continous variables."""
    median = df[column].median()
    df.fillna({column: median}, inplace=True)


def dropna_in_column(df, column):
    """Drop all rows that have a missing value in the specified column."""
    df.dropna(subset=[column], inplace=True)


def handle_null_values(df):
    """Converts any null values in the dataset to their column's mode or median."""

    # As we are not using these columns, we do not need to drop the invalid data anymore
    #dropna_in_column(df, "first_review")
    #dropna_in_column(df, "last_review")
    #dropna_in_column(df, "review_scores_rating")
    #dropna_in_column(df, "host_response_rate")

    fill_na_with_mode(df, "bathrooms")
    fill_na_with_mode(df, "host_identity_verified")
    fill_na_with_mode(df, "bedrooms")
    fill_na_with_mode(df, "beds")
    fill_na_with_median(df, "host_since")


def categorical_to_numeric(df, column, unique_values):
    """Convert each unique value in the column to its index in unique_values."""
    for i, value in enumerate(unique_values):
        df.loc[df[column] == value, column] = i


def convert_columns_to_numeric(df):
    """Convert the values of all columns into numerical values."""

    # Convert all values of super_strict_30 or super_strict_60 in cancellation_policy to strict.
    df.loc[(df["cancellation_policy"] == "super_strict_30") | (df["cancellation_policy"] == "super_strict_60"), "cancellation_policy"] = "strict"
    
    # Convert string values to numeric values
    categorical_to_numeric(df, "cancellation_policy", ["flexible", "moderate", "strict"])
    categorical_to_numeric(df, "host_identity_verified", ["f" , "t"])
    categorical_to_numeric(df, "instant_bookable", ["f", "t"])
    
    # Convert cleaning fee from boolean to numeric
    df["cleaning_fee"] = df["cleaning_fee"].astype(int)

    return df


