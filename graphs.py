"""Each function in this file will plot a graph. Run the file to show all graphs."""
from sklearn.tree import DecisionTreeRegressor

from plotting_helpers import *
import analysis
from train import get_processed_data


def log_price_distribution(df):
    """Plot the distribution of log_price"""
    df["log_price"].plot(kind="hist")
    plt.xlabel("Log Price")
    plt.show()


# The following three functions plot histograms of the distributions of various columns into subplots.
def continous_dist1(df):
    to_plot = ["number_of_reviews", "review_scores_rating", "host_response_rate"]
    plot_histograms(df, to_plot, 2, 2)
    plt.show()

def continous_dist2(df):
    to_plot = ["host_since", "last_review", "first_review"]
    plot_year_bargraphs(df, to_plot, 2, 2)
    plt.show()

def continous_dist3(df):
    to_plot = ["latitude", "longitude"]
    plot_histograms(df, to_plot, 1, 2)
    plt.show()


# The following two functions plot bar graphs that show the distribution of the categorical variables.
def visual_analysis_categorical(df):
    df.groupby("cancellation_policy")["cancellation_policy"].count().plot(kind="bar")
    plt.tight_layout(h_pad=5.0)
    plt.show()

def visual_analysis_categorical2(df):
    plot_bar_graphs(df, ["host_has_profile_pic", "host_identity_verified", "instant_bookable", "cleaning_fee"], 2, 2)
    plt.show()

"""
def outlier_removal_analysis_accommodates(df):
    outlier_removal_analysis_bar(df, "accommodates", suptitle="Outlier Removal Accommodates")


def outlier_removal_analysis_review_scores_rating(df):
    outlier_removal_analysis_hist(df, "review_scores_rating", suptitle="Outlier Removal Review Scores Rating")


def outlier_removal_analysis_host_response_rate(df):
    outlier_removal_analysis_hist(df, "host_response_rate", suptitle="Outlier Removal Analysis Host Response Rate")
"""

def outlier_removal_graphs(df):
    outlier_removal_analysis_bar(df, "bathrooms", suptitle="Outlier Removal Bathrooms")
    outlier_removal_analysis_bar(df, "beds", suptitle="Outlier Removal Beds")
    outlier_removal_analysis_bar(df, "bedrooms", suptitle="Outlier Removal Bedrooms")
    outlier_removal_analysis_bar(df, "accommodates", suptitle="Outlier Removal Accommodates")
    outlier_removal_analysis_hist(df, "review_scores_rating", suptitle="Outlier Removal Review Scores Rating")
    outlier_removal_analysis_hist(df, "host_response_rate", suptitle="Outlier Removal Analysis Host Response Rate")
    


def scatter_host_response_rate(df):
    df.plot.scatter(x="host_response_rate", y="log_price")
    plt.show()


def scatter_first_review(df):
    df.plot.scatter(x="first_review", y="log_price")
    plt.show()


def scatter_last_review(df):
    df.plot.scatter(x="last_review", y="log_price")
    plt.show()


def scatter_review_scores_rating(df):
    df.plot.scatter(x="review_scores_rating", y="log_price")
    plt.show()


def box_plots(df):
    handle_all_outliers(df)
    handle_null_values(df)
    categorical_columns = ["beds", "bedrooms", "bathrooms", "accommodates", "cancellation_policy", "host_identity_verified", "instant_bookable", "cleaning_fee"]
    #fig, plot_canvas = plt.subplots(nrows=7, ncols=1)
    for i, column in enumerate(categorical_columns):
        df.boxplot(column="log_price", by=column, figsize=(10,5))
        plt.show()


def plot_feature_importance():
    """Plot the 10 most important features using DecisionTreeRegressor"""
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    predictors = analysis.get_correlated_predictors()
    df, X, y = get_processed_data(predictors=predictors)
    reg_model = DecisionTreeRegressor(max_depth=5,criterion='friedman_mse')
    XGB = reg_model.fit(X, y)
    feature_importances = pd.Series(XGB.feature_importances_, index=predictors)
    feature_importances.nlargest(10).plot(kind='barh')
    plt.show()


def show_all_plots(df):
    """
    Show all plots one after the other. The next plot will be shown when the window is closed. 
    
    To stop showing more plots, force quit the python program.
    """
    handle_null_values(df)
    log_price_distribution(df)
    continous_dist1(df)
    continous_dist2(df)
    continous_dist3(df)
    visual_analysis_categorical(df)
    visual_analysis_categorical2(df)
    outlier_removal_graphs(df)
    scatter_host_response_rate(df)
    scatter_first_review(df)
    scatter_last_review(df)
    scatter_review_scores_rating(df)
    box_plots(df)
    plot_feature_importance()


if __name__ == "__main__":
    df = get_formatted_dataframe()
    
    show_all_plots(df)
    