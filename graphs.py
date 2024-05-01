
from plotting_helpers import *


def log_price_distribution(df):
    df["log_price"].plot(kind="hist")
    plt.xlabel("Log Price")
    plt.show()


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


def visual_analysis_categorical(df):
    df.groupby("cancellation_policy")["cancellation_policy"].count().plot(kind="bar")
    plt.tight_layout(h_pad=5.0)
    plt.show()


def outlier_removal_analysis_accommodates(df):
    outlier_removal_analysis_bar(df, "accommodates", suptitle="Outlier Removal Accommodates")


def outlier_removal_analysis_review_scores_rating(df):
    outlier_removal_analysis_hist(df, "review_scores_rating", suptitle="Outlier Removal Review Scores Rating")


def outlier_removal_analysis_host_response_rate(df):
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


def box_plot(df, column):
   
    df.boxplot(column="log_price", by=column)
    #plt.suptitle(f"log_price vs {column}")
    plt.show()


if __name__ == "__main__":
    df = get_formatted_dataframe()
    #outlier_removal_analysis_accommodates(df)
    #outlier_removal_analysis_review_scores_rating(df)
    outlier_removal_analysis_host_response_rate(df)
    #log_price_distribution(df)
    #continous_dist3(df)
    #visual_analysis_categorical(df)
    #scatter_last_review(df)
    #scatter_review_scores_rating(df)
    #box_plot(df, "beds")