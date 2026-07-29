"""Grouped data-profiling driver.

:class:`DataProfiler` adds segmentation on top of the functions in
:mod:`DataProfiling_utils`: a dataset can be split by a column, by explicit
intervals, by equal-width bins or by quantiles, and every analysis is then run
per segment and written to its own folder.

The pandas path delegates to the vectorised engine in the :mod:`profiling`
package. The Spark branches are kept for the Databricks notebook workflow and
are only imported when running inside a Databricks runtime.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from DataProfiling_utils import (
    describe_data,
    merge_one_hot_encoded_columns,
    missing_count,
    perform_correlation_analysis,
    plot_histograms_of_data_v3,
    plot_outliers_v3,
)

if os.environ.get("DATABRICKS_RUNTIME_VERSION", None) is not None:
    from pyspark.ml.feature import Bucketizer, QuantileDiscretizer
    from pyspark.sql import functions as F
    from pyspark.sql.functions import col, udf

    import pyspark.pandas as ps

__all__ = ["DataProfiler"]


class DataProfiler:
    """Run a full profiling pass over a dataset, optionally per segment.

    Attributes
    ----------
    save_folder_name : str
        Folder that receives the results of profiling.

    Methods
    -------
    describe_data
        Statistical description of each column.
    missing_analysis
        Counts and percentages of missing values.
    correlation_analysis
        Correlation matrices plus the feature pairs above a threshold.
    plot_hist
        Histograms of the data columns.
    plot_outliers
        Box plots of the data columns.
    perform_data_analysis
        Runs the selected analyses in one call.
    """

    #: Column names of the most recent dataset, excluding the tag column added
    #: when a dict of frames is concatenated. Set by ``_concat_n_datasets`` and
    #: deliberately kept on the class, which is where the classmethod writes it.
    feat_names: list = []

    def __init__(self, save_folder_name="Results_of_DataProfiling"):
        """Create a profiler that writes into ``save_folder_name``.

        Parameters
        ----------
        save_folder_name : str, optional
            Folder used for saving results. Created if it does not exist.
        """
        self._save_folder = save_folder_name

        if save_folder_name is not None and not os.path.exists(self._save_folder):
            os.makedirs(self._save_folder, exist_ok=True)

    @classmethod
    def _get_groups(cls, data, groupby_series=None, bins=None, n_bins=None, n_quantiles=None):
        """Split the data and return the row indices of each group.

        With ``groupby_series=None`` the whole dataset is returned as the
        single group ``{"All": data.index}``.

        Parameters
        ----------
        data : DataFrame
            Input data.
        groupby_series : str or Series, optional
            Column name or series to group on.
        bins : list of tuples, optional
            Explicit intervals to cut the grouping series into, e.g.
            ``[(1, 20), (20, 50)]``.
        n_bins : int, optional
            Number of equal-width bins. Bin occupancy is not guaranteed to be
            even and a bin may end up empty.
        n_quantiles : int, optional
            Number of equal-sized bins, so each group holds a similar number of
            rows.

        Returns
        -------
        groups : dict
            ``{bin: index}`` for every group.
        grouping_var : str
            Name of the column that was grouped on.
        """

        if "spark" in str(type(data)):
            if "sql" in str(type(data)):  # convert pyspark.sql to pyspark.pandas
                data = data.pandas_api()

            if isinstance(groupby_series, str):
                groupby_series = data[groupby_series]

            groups = {}

            if groupby_series is None:
                groups = {"All": data.index}
                grouping_var = "Data"

            else:
                if isinstance(groupby_series, (np.ndarray, np.generic)):
                    groupby_series = ps.Series(groupby_series).rename("Grouped_variable")

                grouping_var = groupby_series.name
                groupby_series = groupby_series.to_frame()  # convert series to df

                if n_bins is None and bins is None and n_quantiles is None:
                    # convert to pyspark.pandas so groupby exposes group indices
                    if "pyspark.sql" in str(type(groupby_series)):
                        groupby_series = groupby_series.pandas_api()

                    for i in groupby_series[grouping_var].unique().dropna().to_list():
                        groups[i] = groupby_series.groupby(grouping_var).get_group(i).index

                elif bins is not None:
                    # QuantileDiscretizer lives on pyspark.sql, not pyspark.pandas
                    if "pyspark.pandas" in str(type(groupby_series)):
                        groupby_series = groupby_series.to_spark()

                    splits = set()  # flatten the tuples into a sorted split list
                    for bin_range in bins:
                        splits.add(float(bin_range[0]))
                        splits.add(float(bin_range[1]))
                    splits = sorted(splits)

                    # map the bucketizer's numeric output back to intervals
                    bins_dict = {None: None, np.nan: np.nan}
                    for i in range(len(splits) - 1):
                        bins_dict[i] = str(pd.Interval(left=splits[i], right=splits[i + 1], closed="right"))

                    def map_func2(key, dic=bins_dict):
                        return dic.get(key, None)

                    map_udf2 = udf(map_func2)

                    bucketizer = Bucketizer(
                        splits=splits, inputCol=grouping_var, outputCol="bucket", handleInvalid="keep"
                    )

                    groupby_series = bucketizer.transform(groupby_series)
                    groupby_series = groupby_series.withColumn("bucket", map_udf2(col("bucket")))
                    groupby_series = groupby_series.pandas_api()

                    for i in groupby_series["bucket"].unique().dropna().to_list():
                        groups[i] = groupby_series.groupby("bucket").get_group(i).index

                elif n_bins is not None:
                    if "pyspark.pandas" in str(type(groupby_series)):
                        groupby_series = groupby_series.to_spark()

                    min_val, max_val = groupby_series.select(F.min(grouping_var), F.max(grouping_var)).first()
                    min_val = min_val - 0.001 * abs(min_val)
                    bin_size = (max_val - min_val) / n_bins

                    bins_list = [min_val] + [min_val + i * bin_size for i in range(1, n_bins)] + [max_val]
                    bins_dict = {None: None, np.nan: np.nan}
                    for i in range(n_bins - 1):
                        bins_dict[i] = str(
                            pd.Interval(
                                left=np.round(bins_list[i], 4),
                                right=np.round(bins_list[i + 1], 4),
                                closed="right",
                            )
                        )

                    def map_func1(key, dic=bins_dict):
                        return dic.get(key, None)

                    map_udf1 = udf(map_func1)

                    splits = (
                        [float("-inf")] + [min_val + i * bin_size for i in range(1, n_bins)] + [float("inf")]
                    )

                    bucketizer = Bucketizer(
                        splits=splits, inputCol=grouping_var, outputCol="bucket", handleInvalid="keep"
                    )

                    groupby_series = bucketizer.transform(groupby_series)
                    groupby_series = groupby_series.withColumn("bucket", map_udf1(col("bucket")))
                    groupby_series = groupby_series.pandas_api()

                    for i in groupby_series["bucket"].unique().dropna().to_list():
                        groups[i] = groupby_series.groupby("bucket").get_group(i).index

                elif n_quantiles is not None:
                    if "pyspark.pandas" in str(type(groupby_series)):
                        groupby_series = groupby_series.to_spark()

                    quantile_discretizer = QuantileDiscretizer(
                        numBuckets=n_quantiles,
                        inputCol=grouping_var,
                        outputCol="bucket",
                        handleInvalid="keep",
                        relativeError=0.0001,
                    )

                    quantile_discretizer_model = quantile_discretizer.fit(groupby_series)
                    groupby_series = quantile_discretizer_model.transform(groupby_series)

                    # recover the split points so the buckets can be labelled
                    min_val, max_val = groupby_series.select(F.min(grouping_var), F.max(grouping_var)).first()
                    splits = [min_val] + sorted(quantile_discretizer_model.getSplits()[1:-1]) + [max_val]

                    bins_dict = {None: None, np.nan: np.nan}
                    for i in range(len(splits) - 1):
                        bins_dict[i] = str(pd.Interval(left=splits[i], right=splits[i + 1], closed="left"))
                    # the last bin is closed on both sides
                    bins_dict[len(splits) - 2] = bins_dict[len(splits) - 2].replace(")", "]")

                    def map_func3(key, dic=bins_dict):
                        return dic.get(key, None)

                    map_udf3 = udf(map_func3)

                    groupby_series = groupby_series.withColumn("bucket", map_udf3(col("bucket")))
                    groupby_series = groupby_series.pandas_api()

                    for i in groupby_series["bucket"].unique().dropna().to_list():
                        groups[i] = groupby_series.groupby("bucket").get_group(i).index

        else:
            gb = None

            if isinstance(groupby_series, str):
                groupby_series = data[groupby_series]

            if groupby_series is None:
                groups = {"All": data.index}
                grouping_var = "Data"

            else:
                if isinstance(groupby_series, (np.ndarray, np.generic)):
                    groupby_series = pd.Series(groupby_series).rename("Grouped_variable")

                grouping_var = groupby_series.name

                if n_bins is None and bins is None and n_quantiles is None:
                    gb = groupby_series.groupby(groupby_series, observed=True)
                elif bins is not None:
                    bins = pd.IntervalIndex.from_tuples(bins)
                    gb = groupby_series.groupby(pd.cut(groupby_series, bins=bins), observed=True)
                elif n_bins is not None:
                    gb = groupby_series.groupby(pd.cut(groupby_series, bins=n_bins), observed=True)
                elif n_quantiles is not None:
                    gb = groupby_series.groupby(pd.qcut(groupby_series, q=n_quantiles), observed=True)

                groups = gb.groups

        cls.groups = groups
        cls.grouping_var = grouping_var

        return groups, grouping_var

    @classmethod
    def _concat_n_datasets(cls, data, joined_col_name="Dataset"):
        """Concatenate a dict of dataframes, tagging each with its key.

        Parameters
        ----------
        data : DataFrame or dict
            A single frame, or ``{'Train': Xy_train, 'Test': Xy_test}``.
        joined_col_name : str, optional
            Name of the column that records which frame a row came from.

        Returns
        -------
        DataFrame
        """
        if isinstance(data, dict):
            key_list = list(data.keys())
            if "spark" in str(type(data[key_list[0]])):
                combined = ps.DataFrame({})
                for key in key_list:
                    if "spark.sql" in str(type(data[key])):
                        data[key] = data[key].pandas_api()

                    temp_df = data[key].copy()
                    temp_df[joined_col_name] = key
                    combined = ps.concat([combined, temp_df], ignore_index=True)
            else:
                combined = pd.concat(data.values(), ignore_index=True)
                combined[joined_col_name] = np.repeat(
                    list(data.keys()), [x.shape[0] for x in data.values()]
                )

            feat_names = list(combined.columns)
            feat_names.remove(joined_col_name)
            cls.feat_names = feat_names
            return combined

        cls.feat_names = list(data.columns)
        return data

    @staticmethod
    def _create_saving_folders(main_save_folder, groups=None, grouping_var=None):
        """Create the output folder and one sub-folder per group.

        Sub-folders are named ``{grouping_var}={group_name}``, for example
        ``Results_of_DataProfiling/Age=(10, 26.5]`` when grouping by age, or
        ``Results_of_DataProfiling/Data=All`` when not grouping at all.

        Parameters
        ----------
        main_save_folder : str
            Root output folder.
        groups : dict, optional
            Groups returned by :meth:`_get_groups`.
        grouping_var : str, optional
            Name of the column used for grouping.
        """
        os.makedirs(main_save_folder, exist_ok=True)
        for group_name, _ in (groups or {}).items():
            os.makedirs(f"{main_save_folder}/{grouping_var}={group_name}", exist_ok=True)

    def _iter_groups(self, data, groupby, bins, n_bins, n_quantiles, save_results):
        """Yield ``(group_label, group_frame, saving_path)`` for each segment."""
        groups, grouping_var = self._get_groups(
            data, groupby, bins=bins, n_bins=n_bins, n_quantiles=n_quantiles
        )
        if save_results:
            self._create_saving_folders(self._save_folder, groups, grouping_var)

        is_spark = "spark" in str(type(data))
        for group_name, indices in groups.items():
            label = f"{grouping_var}={group_name}"
            saving_path = f"{self._save_folder}/{label}" if save_results else None
            subset = data.iloc[indices.to_numpy()] if is_spark else data.loc[indices]
            yield label, subset, saving_path

    def describe_data(
        self,
        data,
        selected_features=None,
        numeric_only=True,
        round_columns=None,
        drop_columns=None,
        groupby=None,
        bins=None,
        n_bins=None,
        n_quantiles=None,
        save_results=True,
    ):
        """Statistical description of every column, per group.

        Parameters
        ----------
        data : DataFrame or dict
            A single frame, or ``{'Train': Xy_train, 'Test': Xy_test}``.
        selected_features : list, optional
            Columns to analyse. ``None`` analyses all of them.
        numeric_only : bool
            Restrict the profile to numeric columns.
        round_columns : dict, optional
            ``{column: decimals}`` string formatting for the output columns.
        drop_columns : list, optional
            Output columns to omit.
        groupby : str or Series, optional
            Column or series to segment on.
        bins, n_bins, n_quantiles
            Segmentation controls, as described in :meth:`_get_groups`.
        save_results : bool
            Write the results to the save folder.

        Returns
        -------
        DataFrame
            The description of the last group processed.
        """
        data = self._concat_n_datasets(data)
        if "spark.sql" in str(type(data)):
            data = data.pandas_api()

        data = data[selected_features] if selected_features is not None else data[self.feat_names]

        result = pd.DataFrame()
        for _, subset, saving_path in self._iter_groups(
            data, groupby, bins, n_bins, n_quantiles, save_results
        ):
            result = describe_data(
                subset,
                numeric_only=numeric_only,
                round_columns=round_columns,
                drop_columns=drop_columns,
                saving_path=saving_path,
            )
        return result

    def missing_analysis(
        self,
        data,
        selected_features=None,
        joined_col_name="DataSet",
        groupby=None,
        bins=None,
        n_bins=None,
        n_quantiles=None,
        save_results=True,
    ):
        """Missing-value counts and percentages, per group.

        Parameters
        ----------
        data : DataFrame or dict
            A single frame, or ``{'Train': Xy_train, 'Test': Xy_test}``.
        selected_features : list, optional
            Columns to analyse. ``None`` analyses all of them.
        joined_col_name : str, optional
            Name of the column tagging which frame a row came from, when
            ``data`` is a dict.
        groupby, bins, n_bins, n_quantiles
            Segmentation controls, as described in :meth:`_get_groups`.
        save_results : bool
            Write the results to the save folder.

        Returns
        -------
        DataFrame
            The missing-value analysis of the last group processed.
        """
        data = self._concat_n_datasets(data, joined_col_name)
        if "spark.sql" in str(type(data)):
            data = data.pandas_api()

        data = data[selected_features] if selected_features is not None else data[self.feat_names]

        result = pd.DataFrame()
        for _, subset, saving_path in self._iter_groups(
            data, groupby, bins, n_bins, n_quantiles, save_results
        ):
            result = missing_count(subset, sort=True, saving_path=saving_path)
        return result

    def correlation_analysis(
        self,
        data,
        joined_col_name="DataSet",
        methods=("Pearson", "Spearman"),
        top_corr_thr=0.8,
        groupby=None,
        bins=None,
        n_bins=None,
        n_quantiles=None,
        save_results=True,
    ):
        """Correlation analysis per group.

        Parameters
        ----------
        data : DataFrame or dict
            A single frame, or ``{'Train': Xy_train, 'Test': Xy_test}``.
        joined_col_name : str, optional
            Name of the column tagging which frame a row came from, when
            ``data`` is a dict.
        methods : tuple
            Any of ``'Pearson'``, ``'Spearman'``, ``'Kendall'``.
        top_corr_thr : float
            Pairs with an absolute coefficient above this are reported.
        groupby, bins, n_bins, n_quantiles
            Segmentation controls, as described in :meth:`_get_groups`.
        save_results : bool
            Write the heatmaps and tables to the save folder.

        Returns
        -------
        DataFrame
            Ranked feature pairs above the threshold for the last group.
        """
        data = self._concat_n_datasets(data, joined_col_name)
        if "spark.sql" in str(type(data)):
            data = data.pandas_api()
        data = data[self.feat_names]

        result = None
        for _, subset, saving_path in self._iter_groups(
            data, groupby, bins, n_bins, n_quantiles, save_results
        ):
            result = perform_correlation_analysis(subset, methods, top_corr_thr, saving_path)
        return result

    def plot_hist(
        self,
        data,
        selected_features=None,
        joined_col_name="DataSet",
        save_results=True,
        groupby=None,
        bins=None,
        n_bins=None,
        n_quantiles=None,
        **plot_kwargs,
    ):
        """Histograms of the data columns, per group.

        Parameters
        ----------
        data : DataFrame or dict
            A single frame, or ``{'Train': Xy_train, 'Test': Xy_test}``.
        selected_features : list, optional
            Columns to plot. ``None`` plots all of them.
        joined_col_name : str, optional
            Name of the column tagging which frame a row came from, when
            ``data`` is a dict.
        save_results : bool
            Write the plots to the save folder.
        groupby, bins, n_bins, n_quantiles
            Segmentation controls, as described in :meth:`_get_groups`.
        **plot_kwargs
            ``n_bins`` (default 40), ``n_cols`` (default 3) and ``stat``.
        """
        data = self._concat_n_datasets(data, joined_col_name)

        if "spark.sql" in str(type(data)):
            data = data.pandas_api()

        if len(self.feat_names) < data.shape[1] and selected_features is None:
            selected_features = self.feat_names

        for label, subset, saving_path in self._iter_groups(
            data, groupby, bins, n_bins, n_quantiles, save_results
        ):
            plot_histograms_of_data_v3(subset, selected_features, saving_path, label, plot_kwargs)

    def plot_outliers(
        self,
        data,
        selected_features=None,
        joined_col_name="DataSet",
        save_results=True,
        groupby=None,
        bins=None,
        n_bins=None,
        n_quantiles=None,
        **plot_kwargs,
    ):
        """Box plots of the data columns, per group.

        Parameters
        ----------
        data : DataFrame or dict
            A single frame, or ``{'Train': Xy_train, 'Test': Xy_test}``.
        selected_features : list, optional
            Columns to plot. ``None`` plots every numeric column.
        joined_col_name : str, optional
            Name of the column tagging which frame a row came from, when
            ``data`` is a dict.
        save_results : bool
            Write the plots to the save folder.
        groupby, bins, n_bins, n_quantiles
            Segmentation controls, as described in :meth:`_get_groups`.
        **plot_kwargs
            Accepted for signature compatibility.
        """
        data = self._concat_n_datasets(data, joined_col_name)

        if "spark.sql" in str(type(data)):
            data = data.pandas_api()

        for label, subset, saving_path in self._iter_groups(
            data, groupby, bins, n_bins, n_quantiles, save_results
        ):
            plot_outliers_v3(subset, selected_features, saving_path, label, plot_kwargs)

    def perform_data_analysis(
        self,
        data,
        selected_features=None,
        groupby=None,
        bins=None,
        n_bins=None,
        n_quantiles=None,
        include_describe_data=True,
        include_missing_analysis=False,
        include_correlation_analysis=True,
        correlation_methods=("Pearson", "Spearman"),
        top_corr_thr=0.7,
        include_plot_hist=True,
        plots_hue_col=None,
        include_plot_outliers=True,
    ):
        """Run the selected analyses in one call.

        Parameters
        ----------
        data : DataFrame or dict
            A single frame, or ``{'Train': Xy_train, 'Test': Xy_test}``.
        selected_features : list, optional
            Columns to analyse. ``None`` analyses all of them.
        groupby, bins, n_bins, n_quantiles
            Segmentation controls, as described in :meth:`_get_groups`.
        include_describe_data, include_missing_analysis,
        include_correlation_analysis, include_plot_hist, include_plot_outliers : bool
            Which analyses to run.
        correlation_methods : tuple
            Correlation methods to compute.
        top_corr_thr : float
            Threshold above which feature pairs are reported.
        plots_hue_col : str, optional
            Column used to colour the plots.

        Returns
        -------
        None
            Results are written to the save folder.
        """
        if include_describe_data:
            self.describe_data(
                data,
                selected_features=selected_features,
                save_results=True,
                groupby=groupby,
                bins=bins,
                n_bins=n_bins,
                n_quantiles=n_quantiles,
            )

        if include_missing_analysis:
            self.missing_analysis(
                data,
                selected_features=selected_features,
                joined_col_name="DataSet",
                groupby=groupby,
                bins=bins,
                n_bins=n_bins,
                n_quantiles=n_quantiles,
                save_results=True,
            )

        if include_correlation_analysis:
            self.correlation_analysis(
                data,
                joined_col_name="DataSet",
                methods=correlation_methods,
                top_corr_thr=top_corr_thr,
                groupby=groupby,
                bins=bins,
                n_bins=n_bins,
                n_quantiles=n_quantiles,
                save_results=True,
            )

        if include_plot_hist:
            self.plot_hist(
                data,
                selected_features=selected_features,
                joined_col_name="DataSet",
                save_results=True,
                groupby=groupby,
                bins=bins,
                n_bins=n_bins,
                n_quantiles=n_quantiles,
                hue_col=plots_hue_col,
            )

        if include_plot_outliers:
            self.plot_outliers(
                data,
                selected_features=selected_features,
                joined_col_name="DataSet",
                save_results=True,
                groupby=groupby,
                bins=bins,
                n_bins=n_bins,
                n_quantiles=n_quantiles,
                hue_col=plots_hue_col,
            )
