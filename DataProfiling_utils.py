import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import six
import warnings
from pandas import ExcelWriter

if os.environ.get('DATABRICKS_RUNTIME_VERSION', None) is not None:
    import pyspark.pandas as ps


def df_to_img(data, font_size=10, index_font_size=9, header_color='forestgreen', row_colors=('#E5EEE4', 'w'),
              bbox=(0, 0, 1, 1),
              title=None, wrap=10, index_wrap=10, save=False, save_name="Dataframe.png", show_index=False, round=3,
              **kwargs):
    """
    This function converts the dataframe to image.
    Parameters:
        - data: dataframe
        - font_size: font size of the text in the table. By default it is set to 10
        - header_color: color of the header. By default it is set to forestgreen
        - row_colors: color of the rows. By default it is set to ('#E5EEE4', 'w')
        - bbox: bounding box of the table. By default it is set to (0, 0, 1, 1)
        - title: title of the table. By default it is set to None.
        - wrap: provide number of character after which user wants to wrap the text to next line. User can adjust it based
                on the requirement of the table. By default it is set to 10
        - save: if True, the image will be saved in the current directory. By default it is set to False
        - save_name: name of the image. By default it is set to Dataframe.png
        - show_index: if True, the index of the dataframe will be shown in the table. By default it is set to False
        - round: number of decimal places to round the data. By default it is set to 3
        - **kwargs: additional arguments to be passed to the table. By default it is set to None
    Returns:
        - fig: matplotlib figure object
    """
    if data.empty:
        print("Input table to df_to_img is empty.")
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.text(0.5, 0.5, 'No data to display', horizontalalignment='center', verticalalignment='center', fontsize=12)
        return fig

    # Copy the data to a new dataframe
    df = data.copy()
    # Round the data
    df = df.round(round)
    # Check if index needs to be shown
    if show_index:
        # Insert a new column with index
        df.insert(0, ' ', df.index)

    # Check if wrap is less than 0
    if wrap < 0:
        # Set wrap to default 10 if it is less than 0
        print("Wrap should be more than 0 setting wrap to default 10")
        wrap = 10

    # Change the content of dataframe to string so that wrap can be performed
    df = df.astype('str')
    # Perform wrap in the data frame content
    for i, x in enumerate(df.columns):
        # Perform wrap in columns
        df[x] = df[x].str.wrap(wrap)

    # Perform wrap in columns name to adjust the width of the columns based on header
    df.columns = df.columns.str.wrap(index_wrap)

    # Create the matplotlib axis with desired size of the plot
    num_rows, num_cols = df.shape

    # Calculate the column widths based on the maximum content length in each column
    col_widths = [max(len(df.columns[i]), df.iloc[:, i].apply(lambda x: len(str(x)[:wrap])).max()) for i in
                  range(num_cols)]
    for i, width in enumerate(col_widths):
        col_widths[i] = min(col_widths[i], wrap) + 5

    # Calculate the row heights based on the maximum content length in each row
    row_heights = [max([str(cell).count('\n') for cell in row]) + 1 for row in df.values]
    for i, height in enumerate(row_heights):
        row_heights[i] = height * 3 + 4.5

    # Adjust the height of header row
    max_header_height = max([head_name.count('\n') for head_name in df.columns])
    header_height = ((max_header_height + 1) * 3 + 4.5)

    # Calculate the plot size based on the sum of column widths and row heights
    figsize = (sum(col_widths) * 0.1, (sum(row_heights) + header_height) * 0.08)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    # Create the table with the matplotlib axis
    mpl_table = ax.table(cellText=df.values, bbox=bbox, colLabels=df.columns, colWidths=col_widths,
                         rowLoc='center', cellLoc='center', colLoc='center', **kwargs)

    # Adjust the height of individual row
    for i, height in enumerate(row_heights):
        for j in range(df.shape[1]):
            mpl_table._cells[(i + 1, j)].set_height(height)

    for j, col in enumerate(df.columns):
        mpl_table._cells[(0, j)].set_height(header_height)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        # cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.0)
        if k[0] == 0 or k[1] < 0:
            cell.set_text_props(color='white', weight='bold')  # (weight='bold', color='black')
            cell.set_facecolor(header_color)
            cell.set_fontsize(font_size - 2)

        else:
            if show_index:
                if k[1] == 0:
                    cell.set_text_props(color='black', weight='bold')
                    cell.set_fontsize(index_font_size)
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    if title:
        ax.set_title(title, fontsize=10)
    if save:
        fig.savefig(f"{save_name}.png")
    return fig



def describe_data(data, numeric_only=True, round_columns=None, drop_columns=None,
                  saving_path=None,
                  ):
    """
    returns some descriptions for columns such as column type, max, min, average, number and percentage of uniques values, number and percentage of missing values, most frequent element and its percentage. 
    
    Parameters
    ----------
    data : DataFrame object
        input data
    numeric_only:
    round_columns:
    drop_columns:

    saving_path : str, optional
        the name of the folder used for saving the results. (default is None)
        
    
    Returns
    -------
    result : DataFrame object
        the output dataframe

    """
    if numeric_only:
        if 'spark' in str(type(data)):
            numeric_feats = data.select_dtypes(exclude=['object', 'category']).columns
        else:
            numeric_feats = data.select_dtypes(include=np.number).columns
        data = data[numeric_feats]

    feature_names = list(data.columns)
    
    result = pd.DataFrame()
    result['DataType'] = data.dtypes
    if 'spark' not in str(type(data)):
        for feat, dtype in zip(feature_names, data.dtypes):
            if str(dtype) in ['object', 'category']:
                warnings.warn(
                    f"Warning:{feat} is not a numerical feature and the execution time may increase significantly",
                    stacklevel=2)

    result['n_uniques (excl. Nulls)'] = data.nunique(axis=0).to_numpy()

    result['n_Missing'] = (data.shape[0] - data.count()).to_numpy()

    result['Perc. of Missing'] = ((result['n_Missing']) / data.shape[0]).round(3) * 100

    result['n_Zeros'] = (data == 0).sum(axis=0).to_numpy()
    result['Perc. of Zeros'] = (result['n_Zeros'] / data.shape[0]).round(3) * 100

    most_frequent_value_each_column = []
    freq_of_most_frequent_value_each_column = []
    
    for feat in feature_names:
        n_not_null_values = data[feat].count()

        if n_not_null_values == 0:
            most_frequent_value_each_column.append(np.nan)
            freq_of_most_frequent_value_each_column.append(np.nan)

        else:
            unique_values_this_column = (data[feat].value_counts(normalize=True, dropna=False) * 100)
            most_frequent_value_each_column.append(unique_values_this_column.index.to_numpy()[0])
            freq_of_most_frequent_value_each_column.append(unique_values_this_column.iloc[0])
    result['1st Most Freq'] = most_frequent_value_each_column
    result['Perc. of 1st Most Freq'] = np.round(freq_of_most_frequent_value_each_column, 1)

    result['Min'] = data.min(axis=0).to_numpy()  # .round() makes error here!
    result['1%'] = data.quantile(0.01).round(4).to_numpy()
    result['50%'] = data.quantile(0.5).round(4).to_numpy()  # much faster than Median
    result['99%'] = data.quantile(0.99).round(4).to_numpy()  # apply(lambda x: '%.0f' % x)
    result['Max'] = data.max(axis=0).to_numpy()  # .round() makes error here!
    if round_columns is None:
        round_columns = {'1st Most Freq': 1, 'Min': 1, '1%': 0, '50%': 0, '99%': 0, 'Max': 0}
    for col, d in round_columns.items():
        result[col] = result[col].map(f"{{:,.{d}f}}".format).to_numpy()
    if drop_columns is not None:
        result = result.drop(drop_columns, axis=1)
    categorical_feat = (result['DataType'].isin(['object', 'category']))
    result.loc[categorical_feat, 'Max'] = np.nan
    result.loc[categorical_feat, 'Min'] = np.nan

    if saving_path is not None:
        result.to_csv(f"{saving_path}/Data_Description.csv", index=True)
        # dfi.export(result, f'{saving_path}/Data_Description.png', fontsize=30, max_rows=-1)
        result_img = df_to_img(result, show_index=True)
        
        result_img.savefig(f'{saving_path}/Data_Description.png', dpi=400)

    # result2 = pd.DataFrame()
    # result2 = data.to_spark().summary("count", "min", "1%", "50%","99%", 'min',"max")
    # print(type(result2))
    
    return result


def missing_count(data, sort=True, saving_path=None):
    """
    returns a dataframe containing non-zero counts and percentages of missing values in the input data.

    Parameters
    ----------
    data : DataFrame object
        input data

    sort : bool, optional
        if sort is True, then the output will be sorted in descending order. (default is True)

    saving_path : str, optional
        the name of the folder used for saving the results. (default is None)

    Returns
    -------
    missing_counts : DataFrame object
        a dataframe containing the non-zero numbers and percentages of missing values in the input data.
    """
    # Replace common missing value indicators with np.nan

    MissingCounts = (data.shape[0] - data.count()).to_numpy()
    missing_counts = pd.DataFrame({'number_of_missing': MissingCounts}, index=data.columns.to_numpy())
    missing_counts['percentage_of_missing'] = (missing_counts['number_of_missing'] / data.shape[0]).round(4) * 100
    missing_counts_non_zeros = missing_counts[missing_counts['number_of_missing'] > 0].copy()

    if sort:
        missing_counts_non_zeros.sort_values(by='number_of_missing', inplace=True, ascending=False)

    if saving_path is not None:
        missing_counts.to_csv(f"{saving_path}/Missing_analysis.csv", index=True)
        # dfi.export(missing_counts_non_zeros, f'{saving_path}/Missing_analysis.png', fontsize=30, max_rows=-1)
        result_img = df_to_img(missing_counts_non_zeros, show_index=True, wrap=25, font_size=12)
        
        result_img.savefig(f'{saving_path}/Missing_analysis.png', dpi=400)
        
    return missing_counts_non_zeros





def perform_correlation_analysis(data, methods=('Pearson', 'Spearman'), top_corr_thr=0.7, saving_path=None):
    """
    Performs a correlation analysis (Pearson, Kendall, Spearman, phik) on dataframe.
    Also returns and saves the top correlations with absolute value greater than threshold.

    Parameters
    ----------
    data : DataFrame object or dict
        A single dataframe object or a dictionary of some dataframes which will be concatenated.
        e.g., {'Train': Xy_train, 'Test': Xy_test}

    methods : {'pearson', 'kendall', 'spearman', 'phik'}, optional
        Methods of correlation (default is 'pearson' and 'spearman')

    top_corr_thr : float, optional
        A threshold used to determine top correlations with absolute values more than top_corr_thr. (default is 0.7)

    saving_path : str
        The name of the folder used for saving the results.

    Returns
    -------
    top_abs_correlations : DataFrame object
        A dataframe with three columns: the names of two features, and their correlation value
        (which is greater than threshold).

    Plots correlation matrices and saves them if a saving path is provided.
    """

    def get_top_abs_correlations_v2(corr_values):
        keep = np.triu(np.ones(corr_values.shape), k=1).astype('bool').reshape(corr_values.size)
        res = corr_values.abs().unstack()[keep].reset_index()
        res = pd.DataFrame(res).rename(
            columns={'level_0': 'Feature 1', 'level_1': 'Feature 2', 0: f'absolute {method} correlation values'})

        return res.sort_values(by=f'absolute {method} correlation values', ascending=False).reset_index(drop=True)

    top_abs_correlations = None

    data = data.select_dtypes(include=[np.number]) #drop categorical string columns
    # Convert data to numeric, coerce errors to NaN
    data_numeric = data.select_dtypes(include=[np.number]).apply(pd.to_numeric, errors='coerce')
    # Drop rows with any NaN values
    data_numeric = data_numeric.dropna(axis=0, how='any')

    for method in methods:


        corr_vals = None
        method = method.lower()
        if method == 'pearson':
            if 'spark' not in str(type(data)):
                corr_vals = data.corr(method='pearson')
            else:
                corr_vals = data.dropna().corr(method='pearson')
                corr_vals = corr_vals.to_pandas()

        elif method == 'kendall' or method == 'spearman':
            if 'spark' not in str(type(data)):
                from scipy.stats import kendalltau, spearmanr
                cols = data.columns
                n_cols = data.shape[1]
                corr_vals = np.zeros((n_cols, n_cols))
                for i in range(n_cols):
                    for j in range(i, n_cols):
                        if i == j:
                            corr_vals[i, j] = 1
                        else:
                            f1 = data.iloc[:, i].values
                            f2 = data.iloc[:, j].values

                            mask = ~pd.isna(f1) & ~pd.isna(f2)
                            f1 = f1[mask]
                            f2 = f2[mask]

                            if method == 'kendall':
                                corr_val = kendalltau(f1, f2)[0]
                            else:  # method == 'spearman':
                                corr_val = spearmanr(f1, f2)[0]
                            corr_vals[i, j] = corr_val
                            corr_vals[j, i] = corr_val
                corr_vals = pd.DataFrame(corr_vals, columns=cols, index=cols)
            else:
                corr_vals = data.dropna().corr(method=method)
                corr_vals = corr_vals.to_pandas()

        elif method == 'phik':
            import phik
            corr_vals = data.phik_matrix()

        corr_vals = np.round(corr_vals, 3)

        n_feats = corr_vals.shape[1]
        fig_size = (min(12, n_feats * 0.5) * 1.1, min(12, n_feats * 0.5))
        fig, ax = plt.subplots(figsize=fig_size, dpi=500)
        
        sns.heatmap(corr_vals.dropna(how='all').dropna(how='all', axis=1),
                    vmin=-1, vmax=1,
                    annot=False, fmt=".2f", linewidths=.1, cmap="coolwarm", cbar=True, ax=ax)

        if n_feats < 10:
            y = 1
        elif n_feats < 20:
            y = 0.95
        elif n_feats <= 30:
            y = 0.93
        else:
            y = 0.92

        fig.suptitle(f"Feature correlation using the {method.capitalize()} method", fontsize=14, y=y)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
        #display heatmap
        plt.show()
        
        print("corr_vals without Nulls.shape", corr_vals.dropna(how='all').dropna(how='all', axis=1).shape)

        abs_correlations = get_top_abs_correlations_v2(corr_vals)
        top_abs_correlations = abs_correlations[abs_correlations[f'absolute {method} correlation values'] > top_corr_thr]

        if saving_path is not None:
            # Save the correlation matrix plot
            fig.savefig(f"{saving_path}/Correlation_matrix_{method}.jpg", bbox_inches='tight', dpi=500)

            # Save the top absolute correlations as an image
            if not top_abs_correlations.empty:
                result_img = df_to_img(top_abs_correlations, show_index=True)
                result_img.savefig(f'{saving_path}/top_abs_correlations_{method}.png', dpi=400)

            # Write abs_correlations to Excel
            excel_file = f"{saving_path}/absolute_correlations.xlsx"

            # Always overwrite the Excel file to avoid errors
            with ExcelWriter(excel_file, engine='openpyxl', mode='w') as writer:
                abs_correlations.to_excel(writer, sheet_name=method, index=False)

        plt.close(fig)  # Close the figure to free memory

    return top_abs_correlations

def plot_histograms_of_data_v3(data, selected_features, saving_path, group_name, plot_kwargs):
    """
    returns the countplots and histograms for the columns of input data
    
    Parameters
    ----------
    data : DataFrame object or dict
        input data
    
    selected_features : list, optional
        list of columns to be plotted. (default is None, so function will plot all columns)
    
    saving_path : str
        the name of the folder used for saving the results.
    
    group_name : str
        the name of the group of data used for plotting.
    
    **plot_kwargs : Additional keyword arguments, optional 
        stat : {'count', 'frequency', 'probability', 'percent', 'density'}
            Aggregate statistic to compute in each bin. (default is 'probability')
        n_bins : int, optional 
            number of bins. (default is 40)
        low_thr : int (percentile), optional 
            low threshold percentile of the data. e.x. with low_thr = 5 the plotting function does't include first 5% of data in the columns. (default is 1)
        high_thr : int (percentile), optional 
            high threshold percentile of the data. e.x. with high_thr = 90 the plotting function does't include last 10% of data in the columns. (default is 99)
        hue_col : str, optional 
            Semantic variable (column) that is mapped to determine the color of plot elements. (default is None)
        legend : list of strings, optional 
            if False, suppress the legend for semantic variables. e.x. ['Train', 'Test'] (default is None)
        element : {“bars”, “step”, “poly”}, optional 
            Visual representation of the histogram statistic. (default is 'step')
        fill : bool, optional
            If True, fill in the space under the histogram. (default is True)
        n_cols : int, optional
            number of columns in plotting matrix of the subplots (default is 2)
        
        
    Returns
    -------
    returns the plots   
    """

    stat = plot_kwargs.get('stat', 'probability')
    hist_n_bins = plot_kwargs.get('n_bins', 40)
    low_thr = plot_kwargs.get('low_thr', 1)
    high_thr = plot_kwargs.get('high_thr', 99)
    hue_col = plot_kwargs.get('hue_col', None)
    legend = plot_kwargs.get('legend', None)
    element = plot_kwargs.get("element", 'step')
    fill = plot_kwargs.get("fill", False)
    hue_order = plot_kwargs.get('hue_order', None)

    if selected_features is None:
        selected_features = list(data.columns)
    selected_features = list(selected_features)

    if hue_col in selected_features:
        selected_features.remove(hue_col)

    if len(selected_features) == 1:
        n_cols = 1
    else:
        n_cols = plot_kwargs.get('n_cols', 3)
    n_rows = int(np.ceil(len(selected_features) / n_cols))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6 * n_cols, n_rows * 3.2), dpi=500)
    plt.subplots_adjust(hspace=0.22, wspace=0.1)

    if n_rows == 1:
        y = 1
    elif n_rows == 2:
        y = 0.97
    elif n_rows == 3:
        y = 0.94
    elif n_rows == 4:
        y = 0.92
    elif (n_rows >= 5) and (n_rows <= 7):
        y = 0.91
    elif (n_rows >= 8) and (n_rows <= 15):
        y = 0.9
    elif (n_rows >= 15) and (n_rows <= 25):
        y = 0.89
    elif (n_rows >= 26) and (n_rows <= 55):
        y = 0.885
    else:
        y = 0.882
    fig.suptitle(f"Histogram plot: {group_name}", fontsize=14, y=y)  # 0.96 - n_rows * 0.01

    axes = np.ravel(axes, order='C')

    for i, feat in enumerate(selected_features):

        if hue_col is not None:
            feat_values = data[[feat, hue_col]][:].dropna()
        else:
            feat_values = data[[feat]].dropna()
        
        if "spark" in str(type(feat_values)).lower():
            feat_values = feat_values.to_pandas()
        
        if feat_values[feat].dtype.name in ['object', 'category', 'bool']:

            # if hue_col is not None:
            #     feat_values = data[[feat, hue_col]][:].dropna()
            # else:
            #     feat_values = data[[feat]].dropna()

            sns.countplot(data=feat_values, x=feat, ax=axes[i], hue=hue_col)

        else:
            tmp_values = data[feat].dropna() 
            if "spark" in str(type(tmp_values)).lower(): 
                tmp_values =  data[feat].dropna().to_pandas()
        
            thr_low_val = np.percentile(tmp_values, low_thr)
            thr_high_val = np.percentile(tmp_values, high_thr)
            mask = (tmp_values >= thr_low_val) & (tmp_values <= thr_high_val)

            feat_values = feat_values[mask]

            if "spark" in str(type(hue_order)).lower():
                hue_order=hue_order.to_numpy()
            sns.histplot(feat_values, x=feat, kde=False,
                         stat=stat,  # count, probability
                         bins=hist_n_bins,
                         hue=hue_col,
                         hue_order=hue_order,
                         common_norm=False,
                         fill=fill,
                         element=element,
                         ax=axes[i])  #

        sns.despine()
        axes[i].tick_params(labelsize=7)
        if i % n_cols == 0:
            axes[i].set_ylabel(stat, fontsize=9)
        else:
            axes[i].set_ylabel("")

        axes[i].set_xlabel(axes[i].get_xlabel(), fontsize=9)
        if legend is not None:
            axes[i].legend(legend, fontsize=8)

    if saving_path is not None:
        fig.savefig(f"{saving_path}/DataHistograms_{hist_n_bins}bins.jpg", bbox_inches='tight', dpi=500)

    return fig, axes


def plot_outliers_v3(data, selected_features, saving_path, group_name, plot_kwargs):
    """
    returns the boxplots for the columns of input data 
    
    Parameters
    ----------
    data : DataFrame object or dict
        a single dataframe object or a dictionary of some dataframes which will be concatenated. e.x. {'Train':Xy_train,'Test':Xy_test}
 
    selected_features : list, optional
        list of columns to be plotted. (default is None, so function will plot all columns)
    
    saving_path : str
        the name of the folder used for saving the results.
    
    group_name : str
        the name of the group of data used for plotting.
    
    **plot_kwargs : Additional keyword arguments, optional 
        hue_col : str, optional 
            Semantic variable (column) that is mapped to determine the color of plot elements. (default is None)
        hue_order : vector of strings, optional
            Order to plot the categorical levels in, otherwise the levels are inferred from the data objects. 
        n_cols : int, optional
            number of columns in plotting matrix of the subplots (default is 1)

    Returns
    -------
    returns the boxplots  
    """
    hue_col = plot_kwargs.get('hue_col', None)
    hue_order = plot_kwargs.get('hue_order', None)

    if selected_features is None:
        selected_features = [x for x in data.columns if data[x].dtype.name not in ['object', 'category', 'bool']]
    else:
        selected_features = [x for x in selected_features if data[x].dtype.name not in ['object', 'category', 'bool']]

    n_cols = plot_kwargs.get('n_cols', 1)
    n_rows = int(np.ceil(len(selected_features) / n_cols))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(14, 2.5 * n_rows), dpi=500)
    plt.subplots_adjust(hspace=0.45, wspace=0.0)

    if n_rows == 1:
        y_title = 1
    elif n_rows == 2:
        y_title = 0.97
    elif n_rows == 3:
        y_title = 0.94
    elif n_rows == 4:
        y_title = 0.92
    elif (n_rows >= 5) and (n_rows <= 7):
        y_title = 0.91
    elif (n_rows >= 8) and (n_rows <= 15):
        y_title = 0.9
    elif (n_rows >= 15) and (n_rows <= 25):
        y_title = 0.89
    elif (n_rows >= 26) and (n_rows <= 55):
        y_title = 0.885
    else:
        y_title = 0.882

    fig.suptitle(f"Outliers plot: {group_name}", fontsize=12, y=y_title)  # 0.98 - n_rows * 0.01

    # sns.set_theme(style="whitegrid", palette="pastel")

    axes = np.ravel(axes, order='C')

    red_square = dict(markerfacecolor='darkorange', marker='s', markersize=4)
    # green_square = dict(markerfacecolor='seagreen', marker='s', markeredgecolor='None')  # #
    # flierprops = dict(marker='o', markerfacecolor='r', markersize=12, linestyle='none', markeredgecolor='g')

    for i, feat in enumerate(selected_features):
        if hue_col is not None:
            feat_values = data[[feat, hue_col]][:].dropna()
        else:
            feat_values = data[[feat]].dropna()

        if "spark" in str(type(feat_values)).lower():
            feat_values = feat_values.to_pandas()

        sns.boxplot(data=feat_values, y=hue_col, x=feat, ax=axes[i], linewidth=1, flierprops=red_square, orient='h')
        sns.despine()  # offset=1
        axes[i].set_ylabel('')
        # axes[i].set_xlabel('')
        axes[i].set_xlabel(feat, fontsize=10, rotation=0)
        # axes[i].set_title(feat, fontsize=10, rotation=0)
        axes[i].tick_params(labelsize=9)
        plt.setp(axes[i].get_xticklabels(), fontsize=8)

    if saving_path is not None:
        fig.savefig(f"{saving_path}/Outliers.jpg", bbox_inches='tight', dpi=500)

    return fig, axes


def merge_one_hot_encoded_columns(df, OHE_features= 'Auto', dummy_separator="__"):
    """
    Merges features of a dataframe that has some One-Hot-Encoded features.
    
    Parameters
    ----------
    df: dataframe object
        input data
    
    OHE_features: dict, optional
        One-Hot-Encoded features in a dictionary format. e.x. {'feat1': ['feat1_a', 'feat1_b'], 'feat2': ['feat2_a', 'feat2_b', 'feat2_c'] }
    
    dummy_separator : str, optional
        the separator used in the name of one-hot encoded columns. (default is "__")
        
    Returns
    -------
    df : dataframe object
        a dataframe without one-hot encoded columns.
    """

    def _get_OHE_dict(OHE_feats):
        """
        Create One hot encoded feature dictionary if from a list given by user.
        """
        ohe_dict = dict.fromkeys(OHE_feats)
        cols = list(df.columns)

        for ohe_feat in ohe_dict.keys():
            ohe_dict[ohe_feat] = [x for x in cols if ohe_feat in x]
        return ohe_dict

    if isinstance(OHE_features, list):
        OHE_features = _get_OHE_dict(OHE_features)

    if str(OHE_features).lower() == "auto":
        OHE_features = dict()
        for s in df.columns:
            ind = s.rfind(dummy_separator)
            if ind != -1:
                original_feat = s[:ind]
                if original_feat not in OHE_features:
                    OHE_features[original_feat] = [s]
                else:
                    OHE_features[original_feat].append(s)

    for merged_feat, dummy_features in OHE_features.items():
        if len(dummy_features) > 1:
            ind = df.columns.get_loc(dummy_features[0])
            s = df[dummy_features].idxmax(axis=1, skipna=True)
            df.insert(ind, merged_feat, s)
            df = df.drop(dummy_features, axis=1)

    return df

    

