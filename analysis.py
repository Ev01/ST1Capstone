from handle_data import *


def print_correlation_analysis(df):
    correlation_data = df[["host_response_rate", "first_review", "last_review", "review_scores_rating", "log_price"]].corr()["log_price"]
    print(correlation_data)





if __name__ == "__main__":
    df = get_formatted_dataframe()
    print_correlation_analysis(df)