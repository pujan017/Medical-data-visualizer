import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df["overweight"] = np.where(
    df["weight"] / (df["height"] * df["height"]) * 10000 > 25,
    1,
    0,
)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df["cholesterol"] = np.where(df["cholesterol"] == 1, 0, 1)
df["gluc"] = np.where(df["gluc"] == 1, 0, 1)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt`
    vars = sorted(
        ["cholesterol", "gluc", "smoke", "alco", "active", "overweight"]
    )
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=vars,
    )
    df_cat = df_cat.value_counts().reset_index(name="total")

    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(
        data=df_cat,
        x="variable",
        y="total",
        hue="value",
        col="cardio",
        kind="bar",
        order=vars,
    )
    fig.set_ylabels("total")
    fig.set_xlabels("variable")
    fig = fig.fig
    
    return fig

def draw_heat_map():
    # Clean the data
    clean = df['ap_lo'] >= df['ap_hi']
    clear = df['height'] < df['height'].quantile(0.025)
    cut = df['height'] > df['height'].quantile(0.975)
    safe = df['weight'] < df['weight'].quantile(0.025)
    neat = df['weight'] > df['weight'].quantile(0.975)

    df_heat = df.drop(index=df[clean | cut | clear | safe | neat].index)

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Define the order of columns for the heatmap
    heatmap_order = ['id', 'age', 'sex', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio', 'overweight']
    corr = corr.reindex(index=heatmap_order, columns=heatmap_order)

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Draw the heatmap with 'sns.heatmap()'
    fig, ax = plt.subplots(figsize=(12, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.4, square=True, fmt=".1f", annot=True, ax=ax)

    # Do not modify the next two lines
    # fig.savefig('heatmap.png')
    return fig

