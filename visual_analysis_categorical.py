import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")
# Remove any duplicate rows in the dataset
df.drop_duplicates(inplace=True)

#to_plot = ["bedrooms", "bathrooms", "beds", "accommodates"]
#to_plot = ["host_has_profile_pic", "host_identity_verified", "instant_bookable", "cleaning_fee"]

def plot_bar_graphs(to_plot, rows, columns):
    for i, col_name in enumerate(to_plot):
        plt.subplot(rows, columns, i+1)
        df.groupby(col_name)[col_name].count().plot(kind="bar")
        #plt.title(col_name)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)

#plot_bar_graphs(to_plot, 2, 2)
df.groupby("cancellation_policy")["cancellation_policy"].count().plot(kind="bar")
plt.tight_layout(h_pad=5.0)
plt.show()