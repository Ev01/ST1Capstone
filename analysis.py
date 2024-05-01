from scipy.stats import f_oneway

from handle_data import *


def print_correlation_analysis(df):
    correlation_data = df[["host_response_rate", "first_review", "last_review", "review_scores_rating", "log_price"]].corr()["log_price"]
    print(correlation_data)


def anova_test(df, target_variable, categorical_predictors):

    # Creating an empty list of final selected predictors
    selected_predictors = []
    #print('##### ANOVA Results ##### \n')
    for predictor in categorical_predictors:
        category_groups = df.groupby(predictor)[target_variable].apply(list)
        anova_results = f_oneway(*category_groups)
        # If the ANOVA P-Value is <0.05, that means we reject H0
        p_value = anova_results[1]
        if p_value < 0.05:
            print(predictor, 'is correlated with', target_variable, '| P-Value:', p_value)
            selected_predictors.append(predictor)
        else:
            print(predictor, 'is NOT correlated with', target_variable, '| P-Value:', p_value)
    return selected_predictors


if __name__ == "__main__":
    df = get_formatted_dataframe()
    print_correlation_analysis(df)
    categorical_predictors = ["beds", "bedrooms", "bathrooms", "accommodates", "cancellation_policy", "host_identity_verified", "instant_bookable", "cleaning_fee"]
    anova_test(df, "log_price", categorical_predictors)