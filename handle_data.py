import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np


def get_formatted_dataframe():
    df = pd.read_csv("data.csv")
    # Remove any duplicate rows in the dataset
    df.drop_duplicates(inplace=True)

    # Convert host_response_rate from a percentage inside a string to a float between 0 and 1.
    df["host_response_rate"] = df["host_response_rate"].str.rstrip("%").astype('float') / 100.0

    # Convert date date columns to pandas datetime objects.
    df["host_since"] = pd.to_datetime(df["host_since"])
    df["last_review"] = pd.to_datetime(df["last_review"])
    df["first_review"] = pd.to_datetime(df["first_review"])
    return df



def print_null_percentage(df):
    # isnull converts values to 1 if they are null, or 0 if not null.
    print(df.isnull().mean() * 100)



def get_outlier_upper_limit_sd(series):
    return series.mean() + series.std() * 3

def get_outlier_lower_limit_sd(series):
    return series.mean() - series.std() * 3


def get_outlier_limits_iqr(series):
    q3 = series.quantile(0.75)
    q1 = series.quantile(0.25)
    iqr = q3 - q1
    lower_limit = q1 - iqr * 1.5
    upper_limit = q3 + iqr * 1.5
    return lower_limit, upper_limit


def is_outlier_sd(series):
    """Return a series with each value replaced with True if it is an outlier, or False if it isn't."""
    # Anything outside outside of three standard deviations is an outlier.
    upper_limit = get_outlier_upper_limit_sd(series)
    lower_limit = get_outlier_lower_limit_sd(series)
    print(f"upper limit: {upper_limit}")
    print(f"lower limit: {lower_limit}")

    outliers = (series >= upper_limit) | (series <= lower_limit)
    
    return outliers


def is_outlier_iqr(series):
    lower_limit, upper_limit = get_outlier_limits_iqr(series)
    print(f"upper limit: {upper_limit}")
    print(f"lower limit: {lower_limit}")

    outliers = (series >= upper_limit) | (series <= lower_limit)
    return outliers


def get_outliers_sd(series):
    return series.loc[is_outlier_sd(series)]


def get_outliers_iqr(series):
    return series.loc[is_outlier_iqr(series)]


def outliers_to_limit(df, series_name, convert_to_int=True):
    upper_limit = get_outlier_upper_limit_sd(df[series_name])
    lower_limit = get_outlier_lower_limit_sd(df[series_name])
    if convert_to_int:
        upper_limit = int(upper_limit)
        lower_limit = int(lower_limit)

    df.loc[is_outlier_sd(df[series_name]) & (df[series_name] > df[series_name].mean()), series_name] = upper_limit
    df.loc[is_outlier_sd(df[series_name]) & (df[series_name] < df[series_name].mean()), series_name] = lower_limit



def handle_all_outliers(df):
    outliers_to_limit(df, "host_response_rate")
    outliers_to_limit(df, "review_scores_rating")
    outliers_to_limit(df, "accommodates")


def fill_na_with_mode(df, attribute):
    mode = df[attribute].mode()[0]
    df.fillna({attribute: mode}, inplace=True)

def fill_na_with_median(df, attribute):
    median = df[attribute].median()
    df.fillna({attribute: median}, inplace=True)


def dropna_in_column(df, attribute):
    df.dropna(subset=[attribute], inplace=True)


def handle_null_values(df):
    dropna_in_column(df, "first_review")
    dropna_in_column(df, "last_review")
    dropna_in_column(df, "review_scores_rating")
    dropna_in_column(df, "host_response_rate")

    fill_na_with_mode(df, "bathrooms")
    fill_na_with_mode(df, "host_identity_verified")
    fill_na_with_mode(df, "bedrooms")
    fill_na_with_mode(df, "beds")
    fill_na_with_median(df, "host_since")


