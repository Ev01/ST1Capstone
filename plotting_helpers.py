import matplotlib.pyplot as plt

from handle_data import *

def plot_histograms(df, to_plot, rows, columns):
    """
    Plot histograms of multiple columns of the dataframe as subplots.
    
    Args:
        df: The dataframe containing the columns to plot
        to_plot: A list of strings containing the names of the columns to plot
        rows: The number of rows in the subplot
        columns: The number of columns in the subplot
    """
    for i, col_name in enumerate(to_plot):
        plt.subplot(rows, columns, i+1)
        df[col_name].plot(kind="hist")
        plt.xlabel(col_name)
        #plt.title(col_name)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)


def plot_year_bargraphs(df, to_plot, rows, columns):
    """
    Convert datetime values in each column to years, then plot the distributions of these column in a subplot.

    Args:
        df: The dataframe containing the columns to plot
        to_plot: A list of strings containing the names of the columns to plot
        rows: The number of rows in the subplot
        columns: The number of columns in the subplot
    """
    for i, col_name in enumerate(to_plot):
        plt.subplot(rows, columns, i+1)
        df[col_name].groupby(df[col_name].dt.year).count().plot(kind="bar")
        plt.subplots_adjust(wspace=0.4, hspace=0.4)


def plot_bar_graphs(df, to_plot, rows, columns):
    """
    Plot bar graphs of multiple columns in the dataframe as a subplot.

    Args:
        df: The dataframe containing the columns to plot
        to_plot: A list of strings containing the names of the columns to plot
        rows: The number of rows in the subplot
        columns: The number of columns in the subplot
    """
    for i, col_name in enumerate(to_plot):
        plt.subplot(rows, columns, i+1)
        df.groupby(col_name)[col_name].count().plot(kind="bar")
        #plt.title(col_name)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)


def plot_before_after(series_before, series_after, kind="bar", suptitle=""):
    """Plot a side-by-side comparison of two series. Plots are labelled 'Before' and 'After'"""
    plt.subplot(1, 2, 1)
    series_before.plot(kind=kind)
    plt.xlabel("Before")
    plt.subplot(1, 2, 2)
    series_after.plot(kind=kind)
    plt.xlabel("After")
    plt.suptitle(suptitle)


def outlier_removal_analysis_bar(df, column, suptitle=""):
    """Plot bar graphs that show a comparison of the column before and after its outliers are handled."""
    before = df[column].copy()
    outliers_to_limit(df, column, convert_to_int=True)
    
    plot_before_after(before.value_counts().sort_index(), 
                      df[column].value_counts().sort_index(), 
                      kind="bar", 
                      suptitle=suptitle)
    
    plt.show()


def outlier_removal_analysis_hist(df, column, suptitle=""):
    """Plot bar graphs that show a comparison of the column before and after its outliers are handled."""
    before = df[column].copy()
    outliers_to_limit(df, column, convert_to_int=False)
    plot_before_after(before, 
                      df[column], 
                      kind="hist", 
                      suptitle=suptitle)

    plt.show()

