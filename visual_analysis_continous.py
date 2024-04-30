import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

df = pd.read_csv("data.csv")
# Remove any duplicate rows in the dataset
df.drop_duplicates(inplace=True)

# Convert host_response_rate from a percentage inside a string to a float between 0 and 1.
df["host_response_rate"] = df["host_response_rate"].str.rstrip("%").astype('float') / 100.0

# Convert date date columns to pandas datetime objects.
df["host_since"] = pd.to_datetime(df["host_since"])
df["last_review"] = pd.to_datetime(df["last_review"])
df["first_review"] = pd.to_datetime(df["first_review"])


def plot_histograms(to_plot, rows, columns):
    for i, col_name in enumerate(to_plot):
        plt.subplot(rows, columns, i+1)
        df[col_name].plot(kind="hist")
        plt.xlabel(col_name)
        #plt.title(col_name)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)


def plot_year_bargraphs(to_plot, rows, columns):
    for i, col_name in enumerate(to_plot):
        plt.subplot(rows, columns, i+1)
        df[col_name].groupby(df[col_name].dt.year).count().plot(kind="bar")
        plt.subplots_adjust(wspace=0.4, hspace=0.4)


def continous_dist1():
    to_plot = ["number_of_reviews", "review_scores_rating", "host_response_rate"]
    plot_histograms(to_plot, 2, 2)
    plt.show()


def continous_dist2():
    to_plot = ["host_since", "last_review", "first_review"]
    plot_year_bargraphs(to_plot, 2, 2)
    plt.show()
    #df["host_since"].groupby(df["host_since"].dt.year).count().plot(kind="bar")
    #plt.show()


def continous_dist3():
    to_plot = ["latitude", "longitude"]
    plot_histograms(to_plot, 1, 2)
    plt.show()


def print_null_percentage():
    # isnull converts values to 1 if they are null, or 0 if not null.
    print(df.isnull().mean() * 100)



def get_outlier_upper_limit_sd(series):
    return series.mean() + series.std() * 3

def get_outlier_lower_limit_sd(series):
    series.mean() - series.std() * 3


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


def outliers_to_upper_limit(attribute):
    upper_limit = int(get_outlier_upper_limit_sd(df[attribute]))
    df.loc[is_outlier_sd(df[attribute]), attribute] = upper_limit


print(get_outliers_sd(df["beds"]))


before = df["accommodates"].copy()
outliers_to_upper_limit("accommodates")
#df["accommodates"] = np.log(df["accommodates"])

plt.subplot(1, 2 ,1)
before.value_counts().sort_index().plot(kind="bar")
plt.xlabel("Before")
plt.subplot(1, 2, 2)
df["accommodates"].value_counts().sort_index().plot(kind="bar")
#df["accommodates"].plot(kind="hist", bins=15)
plt.xlabel("After")
plt.show()
