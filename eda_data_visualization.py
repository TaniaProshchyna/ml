import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math


def plot_categorical_vs_target(df, column, hue_column='y'):
    """
    Generates two bar plots for a given categorical column against a target variable ('y'):
    1. Normalized distribution (percentages) of hue_column values within each category of the column.
    2. Absolute counts of column values grouped by hue_column.

    Both plots are ordered consistently by the total count of instances within each
    categorical column value, in descending order.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the categorical column to analyze.
        hue_column (str): The name of the target/hue column ('y' by default).
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

    # Calculate absolute counts of the hue_column for each category in the column
    counts_df = df.groupby(column)[hue_column].value_counts().unstack(fill_value=0)

    # Determine the order of categories by sorting based on the total count of each category
    category_order = df[column].value_counts().sort_values(ascending=False).index.tolist()

    # Plot 1: Normalized Distribution (Percentages of hue_column within each category of column)
    proportions = df.groupby(column)[hue_column].value_counts(normalize=True).unstack() * 100

    # Reindex using the consistent sorting order
    proportions = proportions.reindex(category_order)

    ax0 = proportions.plot.bar(ax=axes[0], rot=45)
    ax0.set_title(f'Subscription Rate by {column} (Percentage within each {column} category)')
    ax0.set_ylabel('Percentage (%)')
    ax0.set_xlabel(column)
    for container in ax0.containers:
        axes[0].bar_label(container, fmt='%.1f%%')

    # Plot 2: Absolute Counts
    # Reindex using the consistent sorting order
    counts_df_reindexed = counts_df.reindex(category_order)

    ax1 = counts_df_reindexed.plot.bar(ax=axes[1], rot=45)
    ax1.set_title(f'Counts of {column} by {hue_column}')
    ax1.set_ylabel('Count')
    ax1.set_xlabel(column)
    for container in ax1.containers:
        axes[1].bar_label(container, fmt='%d')

    plt.tight_layout()
    plt.show()


def plot_continuous_vs_target(df, continuous_vars, target_var='y', palette=None, remove_outliers=False):
    """
    Generates boxplots for continuous variables against a categorical target variable,
    with optional outlier removal and custom palette.

    Args:
        df (pd.DataFrame): The input DataFrame.
        continuous_vars (list): A list of column names for continuous variables.
        target_var (str): The name of the categorical target variable ('y' by default).
        palette (dict, optional): A dictionary mapping target variable categories to colors.
                                 Defaults to None, using seaborn's default palette.
        remove_outliers (bool, optional): If True, outliers will be removed from each
                                        continuous variable before plotting using the IQR method.
    """
    n_cols = 3  # Number of columns for subplots
    n_rows = math.ceil(len(continuous_vars) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    detailed_outlier_report = {}

    for i, col in enumerate(continuous_vars):
        if i < len(axes):
            plot_df_for_col = df.copy() # Create a fresh copy for each column to apply filtering independently

            if remove_outliers:
                detailed_outlier_report[col] = {}
                # Iterate over unique target values to calculate outliers per group
                for target_val in sorted(plot_df_for_col[target_var].unique()):
                    subset = plot_df_for_col[plot_df_for_col[target_var] == target_val][col]
                    if not subset.empty:
                        Q1 = subset.quantile(0.25)
                        Q3 = subset.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        initial_group_count = len(subset)
                        outliers_mask = (plot_df_for_col[target_var] == target_val) & ((plot_df_for_col[col] < lower_bound) | (plot_df_for_col[col] > upper_bound))
                        removed_count_group = plot_df_for_col[outliers_mask].shape[0]
                        percentage_removed = (removed_count_group / initial_group_count * 100) if initial_group_count > 0 else 0

                        detailed_outlier_report[col][target_val] = {
                            'initial_count': initial_group_count,
                            'removed_count': removed_count_group,
                            'percentage_removed': percentage_removed
                        }

                        # Filter out outliers for the specific target group within this column's plot_df_for_col
                        plot_df_for_col = plot_df_for_col[~outliers_mask]


            sns.boxplot(data=plot_df_for_col, x=target_var, y=col, ax=axes[i], showmeans=True, palette=palette, hue=target_var, legend=False)
            axes[i].set_title(f'Distribution of {col} by {target_var}')
            axes[i].set_xlabel(target_var)
            axes[i].set_ylabel(col)

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    if remove_outliers:
        print("\n--- Outlier Removal Report (1.5*IQR method, for plotting purposes) ---")
        for col, target_data in detailed_outlier_report.items():
            print(f"Column: '{col}'")
            for target_val, stats in target_data.items():
                print(f"  - Target '{target_val}': Removed {stats['removed_count']} outliers from {stats['initial_count']} instances ({stats['percentage_removed']:.2f}%).")
        print("------------------------------------------------------------------")
