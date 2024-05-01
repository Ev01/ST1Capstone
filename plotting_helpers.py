import matplotlib.pyplot as plt

from handle_data import *

def plot_histograms(df, to_plot, rows, columns):
    for i, col_name in enumerate(to_plot):
        plt.subplot(rows, columns, i+1)
        df[col_name].plot(kind="hist")
        plt.xlabel(col_name)
        #plt.title(col_name)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)


def plot_year_bargraphs(df, to_plot, rows, columns):
    for i, col_name in enumerate(to_plot):
        plt.subplot(rows, columns, i+1)
        df[col_name].groupby(df[col_name].dt.year).count().plot(kind="bar")
        plt.subplots_adjust(wspace=0.4, hspace=0.4)


def plot_bar_graphs(df, to_plot, rows, columns):
    for i, col_name in enumerate(to_plot):
        plt.subplot(rows, columns, i+1)
        df.groupby(col_name)[col_name].count().plot(kind="bar")
        #plt.title(col_name)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)


def plot_before_after(series_before, series_after, kind="bar", suptitle=""):
    plt.subplot(1, 2, 1)
    series_before.plot(kind=kind)
    plt.xlabel("Before")
    plt.subplot(1, 2, 2)
    series_after.plot(kind=kind)
    plt.xlabel("After")
    plt.suptitle(suptitle)




def outlier_removal_analysis_bar(df, series_name, suptitle=""):
    before = df[series_name].copy()
    outliers_to_limit(df, series_name, convert_to_int=True)
    
    plot_before_after(before.value_counts().sort_index(), 
                      df[series_name].value_counts().sort_index(), 
                      kind="bar", 
                      suptitle=suptitle)
    
    plt.show()


def outlier_removal_analysis_hist(df, series_name, suptitle=""):
    before = df[series_name].copy()
    outliers_to_limit(df, series_name, convert_to_int=False)
    plot_before_after(before, 
                      df[series_name], 
                      kind="hist", 
                      suptitle=suptitle)

    plt.show()

