import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import plotly.express as px

df = pd.read_csv("cleaned-data.csv")

df_categorical = df.iloc[:, :8].copy()
# remove categorical features from original dataframe
df.drop(columns=df_categorical.columns, inplace=True)

cols = df.columns

# Scale the data
df = StandardScaler().fit_transform(df)

# Apply PCA to compute all of the principal components
pca = PCA()
components = pca.fit_transform(df)

# create a new dataframe to store the components
components_df = pd.DataFrame(data = components, columns = ['component-' + str(x) for x in range(components.shape[-1])])

# desired number of components
n = components.shape[-1]


# produce a quick component plot
def test_scatter_plot(comp_A, comp_B):
    plt.scatter(components_df.iloc[:, comp_A], components_df.iloc[:, comp_B])
    plt.show()

# prints the most important feature in each component
def most_important_features():
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(components.shape[-1])]

    # feature names
    most_important_names = [cols[most_important[i]] for i in range(n)]

    dic = {'PC{}'.format(i): most_important_names[i] for i in range(n)}
    df_final = pd.DataFrame(dic.items())

    print(df_final)

# uses plotly to create a scatterplot matrix with the components
def component_matrix(num_components=2, color_map="Item"):
    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    fig = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(num_components),
        color=df_categorical[color_map]
    )
    fig.update_traces(diagonal_visible=False)
    fig.show()

component_matrix()