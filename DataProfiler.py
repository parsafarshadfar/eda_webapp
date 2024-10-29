import os
from DataProfiling_utils import *

if os.environ.get("DATABRICKS_RUNTIME_VERSION", None) is not None:
    from pyspark.ml.feature import Bucketizer, QuantileDiscretizer
    from pyspark.sql import functions as F
    from pyspark.sql.functions import col, when, udf
    import pyspark.pandas as ps



class DataProfiler:
    """
    Class for data profiling

    Attributes
    ----------
    save_folder_name : str
        The name of the folder used for saving results of profiling


    Methods
    -------
    get_groups:
        Groups (segments) the dataframe by one of its columns or a Pandas Series (the same length) and returns the data groups
    
    create_saving_folders:
        creates main directory and its sub directories for saving output files of other functions.
    
    missing_analysis:
        returns a dataframe containing the number and percentages of missing values in the data.
    
    describe_data:
        returns some statistical descriptions for each column of dataframe.
    
    plot_hist:
        plots the histograms of data columns
    
    plot_outliers:
        plots the box plots of data columns
    
    correlation_analysis:
        performs a correlation analysis (Pearson, Phik, Kendall, Spearman) on dataframe.
        It also returns and save the top correlations with absolute value greater than a threshold.
    """

    def __init__(self, save_folder_name="Results_of_DataProfiling"):
        """
        constructs all the necessary attributes for the DataProfiler object.
        
        Parameters
        ----------
        save_folder_name: str, optional
            The name of the folder used for saving results of profiling (default is "Results_of_DataProfiling" )
        
        Returns
        -------
        None
  
        """
        self._save_folder = save_folder_name

        if save_folder_name is not None:
            if not os.path.exists(self._save_folder):
                os.mkdir(self._save_folder)

    @classmethod
    def _get_groups(cls, data, groupby_series=None, bins=None, n_bins=None, n_quantiles=None):
        """
        Groups the data and then returns a dictionary of group indices.
        if groupby_series is None, returns all indices of data as a dictionary with one single entry: {"All":data.index}
        
        Parameters
        ----------
        data : DataFrame object
            input dataframe
        
        groupby_series : str or a Pandas Series, optional
            the column name or a Pandas Series which the get_group function performs grouping on. (default is None)
        
        bins : list of tuples, optional 
            the intervals used by get_group function to cut the groupby_series column and perform grouping. (default is None)
            e.x. [(1, 20), (20, 50)]
            
        n_bins : int, optional 
            the number of equal-width bins used by get_group function to cut the groupby_series column and perform grouping. (default is None)
            (There is no guarantee about the distribution of data in each bin. e.x. a bin may even get empty at the end)
        
        n_quantiles : int, optional 
            the number of equal-sized (not equal-width) bins used by get_group function to cut the groupby_series column 
            and perform grouping. (default is None)
            (distribution of data in the bins is equal)      

        Returns
        -------
        groups : dict
            a dictionary that its keys are bins and its values are list of indexes belonging to each bin.
            e.x. { (1, 20] : [0,1,3,5,6,7], 
                   (20, 50] : [2,4,8,9,10] }
        
        group_var : str
            column name where the get_group function performs grouping on. 
        """

        if 'spark' in str(type(data)):
            if 'sql' in str(type(data)): #convert pyspark.sql to pyspark.pandas
                data = data.pandas_api()

            if type(groupby_series) == str:  
                groupby_series = data[groupby_series]

            groups = {}

            if groupby_series is None:
                groups = {"All": data.index}
                grouping_var = "Data"

            else:
                
                if isinstance(groupby_series, (np.ndarray, np.generic)):
                    groupby_series = ps.Series(groupby_series).rename("Grouped_variable")

                grouping_var = groupby_series.name
                groupby_series = groupby_series.to_frame() #convert series to df


                # if len(np.unique(groupby_series)) < 20:
                if n_bins is None and bins is None and n_quantiles is None:
                    #convert pyspark.sql to pyspark.pandas to have the ability to get index of groups using .index in groupby
                    if 'pyspark.sql' in str(type(groupby_series)):
                        groupby_series = groupby_series.pandas_api() 

                    for i in groupby_series[grouping_var].unique().dropna().to_list():
                        groups[i] = groupby_series.groupby(grouping_var).get_group(i).index
                            
                elif bins is not None:
                    #convert pyspark.pandas to pyspark.sql to use its QuantileDiscretizer function becasue ntile function is not included in pyspark.pandas yet :'
                    if 'pyspark.pandas' in str(type(groupby_series)):
                            groupby_series = groupby_series.to_spark()
                    
                    splits = set() #convert list of tuples to a sorted 1-d list to feed bucketizer
                    for bin_range in bins:
                        splits.add(float(bin_range[0]))
                        splits.add(float(bin_range[1]))
                    splits = sorted(list(splits))
                    
                    # make a mapping dictionary to convert numerical output of bucketizer to intervals
                    bins_dict = {None:None, np.nan:np.nan}
                    for i in range(len(splits) - 1 ):
                        bins_dict[i] = str( pd.Interval(left=splits[i], right=splits[i+1], closed='right')) 
    
                    def map_func2(key, dic = bins_dict):
                        return dic.get(key, None)
                    map_udf2 = udf(map_func2)

                    # Create the bucketizer 
                    bucketizer = Bucketizer(splits=splits, inputCol=grouping_var, outputCol="bucket",handleInvalid='keep')

                    
                    # Apply the bucketizer and create intervals
                    groupby_series = bucketizer.transform(groupby_series)
                    groupby_series = groupby_series.withColumn("bucket", map_udf2(col("bucket")))
                    
                    #convert pyspark.sql to pyspark.pandas to have the ability to get index of groups using .index in groupby
                    groupby_series = groupby_series.pandas_api()

                    for i in groupby_series['bucket'].unique().dropna().to_list():
                        groups[i] = groupby_series.groupby('bucket').get_group(i).index

                elif n_bins is not None:
                    
                    #convert pyspark.pandas to pyspark.sql to use its QuantileDiscretizer function becasue ntile function is not included in pyspark.pandas yet :'
                    if 'pyspark.pandas' in str(type(groupby_series)):
                            groupby_series = groupby_series.to_spark()
                    
                    # df = df.filter(df[grouping_var].isNotNull()) # Filter out null values

                    # Calculate the bin boundaries
                    min_val, max_val = groupby_series.select(F.min(grouping_var), F.max(grouping_var)).first()
                    min_val = min_val - 0.001*abs(min_val)
                    bin_size = (max_val - min_val) / n_bins
                    
                    bins_list = [min_val] + [min_val + i * bin_size for i in range(1, n_bins)] + [max_val]
                    # make a mapping dictionary to convert numerical output of bucketizer to intervals
                    bins_dict = {None:None, np.nan:np.nan}
                    for i in range(n_bins - 1 ):
                        bins_dict[i] = str( pd.Interval(left=np.round(bins_list[i], 4), right=np.round(bins_list[i+1], 4), closed='right') )
                    
                    def map_func1(key, dic = bins_dict):
                        return dic.get(key, None)
                    map_udf1 = udf(map_func1)


                    splits = [float("-inf")] + [min_val + i * bin_size for i in range(1, n_bins)] + [float("inf")]

                    # Create the bucketizer with specified bins aka splits
                    bucketizer = Bucketizer(splits=splits, inputCol=grouping_var, outputCol="bucket",handleInvalid='keep')

                    
                    # Apply the bucketizer and create intervals
                    groupby_series = bucketizer.transform(groupby_series)
                    groupby_series = groupby_series.withColumn("bucket", map_udf1(col("bucket")))
                    
                    #convert pyspark.sql to pyspark.pandas to have the ability to get index of groups using .index in groupby
                    groupby_series = groupby_series.pandas_api() 

                    for i in groupby_series['bucket'].unique().dropna().to_list():
                        groups[i] = groupby_series.groupby('bucket').get_group(i).index
                
                elif n_quantiles is not None:                    

                    #convert pyspark.pandas to pyspark.sql to use its QuantileDiscretizer function becasue ntile function is not included in pyspark.pandas yet :'
                    if 'pyspark.pandas' in str(type(groupby_series)):
                            groupby_series = groupby_series.to_spark()
                            

                    # Create the QuantileDiscretizer transformer
                    quantile_discretizer = QuantileDiscretizer(numBuckets=n_quantiles, inputCol=grouping_var, outputCol="bucket", handleInvalid='keep', relativeError=0.0001)

                    # Fit the transformer on the DataFrame
                    quantile_discretizer_model = quantile_discretizer.fit(groupby_series)
                    
                    # Apply the transformation to the DataFrame
                    groupby_series = quantile_discretizer_model.transform(groupby_series)
                    
                    #get the split points to create mapping dict and udf to make intervals using them
                    min_val, max_val = groupby_series.select(F.min(grouping_var), F.max(grouping_var)).first()
                    splits = [min_val] + sorted(quantile_discretizer_model.getSplits()[1:-1])+ [max_val]
                    
                    #create dict
                    bins_dict = {None:None, np.nan:np.nan}
                    for i in range(len(splits) - 1 ):
                        bins_dict[i] = str( pd.Interval(left=splits[i], right=splits[i+1], closed='left')) 
                    bins_dict[len(splits) - 2] = bins_dict[len(splits) - 2].replace(')',']') #last bin is closed at both sides
                    
                    # def user defined function using dict
                    def map_func3(key, dic = bins_dict):
                        return dic.get(key, None)
                    map_udf3 = udf(map_func3)

                    #apply func on groupby_series to convert numbers to intervals
                    groupby_series = groupby_series.withColumn("bucket", map_udf3(col("bucket")))
                    
                    #convert pyspark.sql to pyspark.pandas to have the ability to get index of groups using .index in groupby
                    groupby_series = groupby_series.pandas_api() 

                    for i in groupby_series['bucket'].unique().dropna().to_list():
                        groups[i] = groupby_series.groupby('bucket').get_group(i).index
   
                

        else:
            gb = None

            if type(groupby_series) == str:  # isinstance(groupby_series, str)
                groupby_series = data[groupby_series]

            if groupby_series is None:
                groups = {"All": data.index}
                grouping_var = "Data"

            else:
                grouping_var = groupby_series.name
                # print(f"Data is grouped by {groupby_series.name}")
                if isinstance(groupby_series, (np.ndarray, np.generic)):
                    groupby_series = pd.Series(groupby_series).rename("Grouped_variable")

                # if len(np.unique(groupby_series)) < 20:
                if n_bins is None and bins is None and n_quantiles is None:
                    gb = groupby_series.groupby(groupby_series)
                elif bins is not None:
                    bins = pd.IntervalIndex.from_tuples(bins)  # generating the bins interval according the groups
                    gb = groupby_series.groupby(pd.cut(groupby_series, bins=bins))
                elif n_bins is not None:
                    gb = groupby_series.groupby(pd.cut(groupby_series, bins=n_bins))
                elif n_quantiles is not None:
                    gb = groupby_series.groupby(pd.qcut(groupby_series, q=n_quantiles))

                groups = gb.groups

        cls.groups = groups
        cls.grouping_var = grouping_var

        return groups, grouping_var

    @classmethod
    def _concat_n_datasets(cls, data, joined_col_name='Dataset'):
        """
        if the input data is dictionary of dataframes, this function returns the concatenation of the dataframes.

        Parameters
        ----------
        data : DataFrame object or dict
            a single dataframe object or a dictionary of some dataframes. e.x. {'Train':Xy_train,'Test':Xy_test}

        joined_col_name : str, optional
            if input data is dict, then this is the name of the new column added to the concatenated dataframes in the data. (default is "DataSet")
            this column would contain the names of the dataframes existing in the data. e.x. 'Train' label or 'Test' label

        Returns
        -------
        returns a dataframe.
        """

        if type(data) in [dict]:
            key_list = list(data.keys()) 
            if 'spark' in str(type(data[key_list[0]])):
                d = ps.DataFrame({})
                for key in key_list:
                    if 'spark.sql' in str(type(data[key])):
                        data[key] = data[key].pandas_api()
                    
                    temp_df = data[key].copy()
                    temp_df[joined_col_name] = key
                    d = ps.concat([d,temp_df], ignore_index=True)
                temp_df = ps.DataFrame({})

            else:
                d = pd.concat(data.values(), ignore_index=True)
                d[joined_col_name] = np.repeat(list(data.keys()), [x.shape[0] for x in data.values()])
            
            feat_names = list(d.columns)
            feat_names.remove(joined_col_name)
            cls.feat_names = feat_names
            return d
        else:
            cls.feat_names = list(data.columns)
            return data

    @staticmethod
    def _create_saving_folders(main_save_folder, groups=None, grouping_var=None):
        """
        creates main folder and its sub folders for saving output files of other functions.
        
        
        Parameters
        ----------
        main_save_folder : str
            name of the main folder
        
        groups : dict, optional 
            a dictionary of bins and their corresponding list if indexes that is derived from get_groups function. (default is None) 
        
        grouping_var: str, optional 
            name of the column used for grouping. (default is None)


        Returns
        -------
        None
        
        \{main_save_folder}\{grouping_var}={group_name}
        e.x. of a main folder and its sub folders when optional inputs are not None:
            ~\Results_of_DataProfiling\Age=(10, 26.5]
        e.x. of a main folder and its sub folder when optional inputs are None:
            ~\Results_of_DataProfiling\Data=All
  
        """

        if not os.path.exists(main_save_folder):
            os.mkdir(main_save_folder)

        for group_name, _ in groups.items():
            if not os.path.exists(f"{main_save_folder}/{grouping_var}={group_name}"):
                os.mkdir(f"{main_save_folder}/{grouping_var}={group_name}")

    def describe_data(self, data, selected_features=None,
                      numeric_only=True, round_columns=None, drop_columns=None,
                      groupby=None, bins=None, n_bins=None, n_quantiles=None,
                      save_results=True):
        """
        returns some descriptions for columns such as column type, max, min, average, number and percentage of uniques values, number and percentage of missing values, most frequent element and its percentage. 
        
        
        Parameters
        ----------
        data : DataFrame object or dict
            a single dataframe object or a dictionary of some dataframes which will be concatenated. e.x. {'Train':Xy_train,'Test':Xy_test}
        
        selected_features : list, optional
            list of columns to be analyzed. (default is None, so all columns would get analyzed)
        drop_columns:
        round_columns:
        numeric_only:

        groupby : str or a Pandas Series, optional 
            the column name or a Pandas Series that the function performs grouping on. (default is None)
        
        bins : list of tuples, optional 
            the intervals used by the function to cut the groupby column and perform grouping. (default is None)
            e.x. [(1, 20), (20, 50)]
            
        n_bins : int, optional 
            the number of equal-width bins used by the function to cut the groupby column and perform grouping. (default is None)
            (There is no guarantee about the distribution of data in each bin. e.x. a bin may even get empty at the end)
        
        n_quantiles : int, optional 
            the number of equal-sized (not equal-width) bins used by the function to cut the groupby column and perform grouping. (default is None)
            
        save_results : bool, optional 
            if save_results is True, the function will save the results. (default is True)
            
            
        Returns
        -------
        returns the output dataframe
        """
        data = self._concat_n_datasets(data)
        if 'spark.sql' in str(type(data)): #convert to pyspark.pandas df, if it is pyspark.sql df
                data = data.pandas_api()

        data = data[selected_features] if selected_features is not None else data[self.feat_names]
        groups, grouping_var = self._get_groups(data, groupby, bins=bins, n_bins=n_bins, n_quantiles=n_quantiles)
        
        self._create_saving_folders(self._save_folder, groups, grouping_var) if save_results else 0
        result = []
        for group_name, indices in self.groups.items():
            saving_path = f"{self._save_folder}/{grouping_var}={str(group_name)}" if save_results else None
            if 'spark' in str(type(data)):
                result = describe_data(data.iloc[indices.to_numpy()] , numeric_only=numeric_only, round_columns=round_columns,
                                   drop_columns=drop_columns,
                                   saving_path=saving_path)
            else: 
                result = describe_data(data.loc[indices] , numeric_only=numeric_only, round_columns=round_columns,
                                   drop_columns=drop_columns,
                                   saving_path=saving_path)
        return result

    def missing_analysis(self, data, selected_features=None, joined_col_name="DataSet",
                         groupby=None, bins=None, n_bins=None, n_quantiles=None,
                         save_results=True):
        """
        returns a dataframe containing non-zero counts and percentages of missing values in the input data, also save the result

        Parameters
        ----------
        data : DataFrame object or dict
            a single dataframe object or a dictionary of some dataframes which will be concatenated. e.x. {'Train':Xy_train,'Test':Xy_test}

        joined_col_name : str, optional
            if input data is dict, then this is the name of the new column added to the concatenated dataframes in the data. (default is "DataSet")
            this column would contain the names of the dataframes existing in the data. e.x. 'Train' label or 'Test' label

        selected_features : list, optional
            list of columns to be analyzed. (default is None, so all columns would get analyzed)

        groupby : str or a Pandas Series, optional
            the column name or a Pandas Series that the function performs grouping on. (default is None)

        bins : list of tuples, optional
            the intervals used by the function to cut the groupby column and perform grouping. (default is None)
            e.x. [(1, 20), (20, 50)]

        n_bins : int, optional
            the number of equal-width bins used by the function to cut the groupby column and perform grouping. (default is None)
            (There is no guarantee about the distribution of data in each bin. e.x. a bin may even get empty at the end)

        n_quantiles : int, optional
            the number of equal-sized (not equal-width) bins used by the function to cut the groupby column and perform grouping. (default is None))

        save_results : bool, optional
            if save_results is True, the function will save the results. (default is True)

        Returns
        -------
        returns the output dataframe containing analysis of missing values

        """
        data = self._concat_n_datasets(data, joined_col_name)
        
        if 'spark.sql' in str(type(data)): #convert to pyspark.pandas df, if it is pyspark.sql df
            data = data.pandas_api()

        data = data[selected_features] if selected_features is not None else data[self.feat_names]

        groups, grouping_var = self._get_groups(data, groupby, bins=bins, n_bins=n_bins, n_quantiles=n_quantiles)
        self._create_saving_folders(self._save_folder, groups, grouping_var) if save_results else 0

        result = []
        for group_name, indices in groups.items():
            saving_path = f"{self._save_folder}/{grouping_var}={str(group_name)}" if save_results else None
            if 'spark' in str(type(data)):
                result = missing_count(data.iloc[indices.to_numpy()], sort=True, saving_path=saving_path)
            else:
                result = missing_count(data.loc[indices], sort=True, saving_path=saving_path)


        return result

    def correlation_analysis(self, data, joined_col_name="DataSet", methods=('Pearson', 'Spearman'), top_corr_thr=0.8,
                             groupby=None, bins=None, n_bins=None, n_quantiles=None,
                             save_results=True, ):
        """
        performs a correlation analysis (Pearson, Kendall, Spearman, phik) on dataframe. also returns and save the top correlations with absolute value greater than threshold

        Parameters
        ----------
        data : DataFrame object or dict
            a single dataframe object or a dictionary of some dataframes which will be concatenated. e.x. {'Train':Xy_train,'Test':Xy_test}

        joined_col_name : str, optional
            if input data is dict, then this is the name of the new column added to the concatenated dataframes in the data. (default is "DataSet")
            this column would contain the names of the dataframes existing in the data. e.x. 'Train' label or 'Test' label

        methods : {'Pearson', 'Kendall', 'Spearman', 'Phik'}, tuple, optional
            method of correlation (default is 'Pearson' and 'Spearman')

        top_corr_thr : float, optional
            a threshold used to determine top correlations with absolute values more than top_corr_thr. (default is 0.8)

        save_results : bool, optional
            if save_results is True, the function will save the plots. (default is True)

        groupby : str or a Pandas Series, optional
            the column name or a Pandas Series that the function performs grouping on. (default is None)

        bins : list of tuples, optional
            the intervals used by the function to cut the groupby column and perform grouping. (default is None)
            e.x. [(1, 20), (20, 50)]

        n_bins : int, optional
            the number of equal-width bins used by the function to cut the groupby column and perform grouping. (default is None)
            (There is no guarantee about the distribution of data in each bin. e.x. a bin may even get empty at the end)

        n_quantiles : int, optional
            the number of equal-sized (not equal-width) bins used by the function to cut the groupby column and perform grouping. (default is None)


        Returns
        -------
        plot correlation matrices
        """
        data = self._concat_n_datasets(data, joined_col_name)
        if 'spark.sql' in str(type(data)): #convert to pyspark.pandas df, if it is pyspark.sql df
            data = data.pandas_api()
        data = data[self.feat_names]

        groups, grouping_var = self._get_groups(data, groupby, bins=bins, n_bins=n_bins, n_quantiles=n_quantiles)
        self._create_saving_folders(self._save_folder, groups, grouping_var) if save_results else 0

        result = []
        for grp_name, indices in groups.items():
            group_name = f"{grouping_var}={str(grp_name)}"
            saving_path = f"{self._save_folder}/{group_name}" if save_results else None
            if 'spark' in str(type(data)):
                result = perform_correlation_analysis(data.iloc[indices.to_numpy()], methods, top_corr_thr, saving_path)
            else:
                result = perform_correlation_analysis(data.loc[indices], methods, top_corr_thr, saving_path)

        return result
    

    def plot_hist(self, data, selected_features=None, joined_col_name="DataSet", save_results=True,
                  groupby=None, bins=None, n_bins=None, n_quantiles=None,
                  **plot_kwargs
                  ):
        """
        plots the histograms of data columns   
        
        Parameters
        ----------
        data : DataFrame object or dict
            a single dataframe object or a dictionary of some dataframes which will be concatenated. e.x. {'Train':Xy_train,'Test':Xy_test}
        
        joined_col_name : str, optional 
            if input data is dict, then this is the name of the new column added to the concatenated dataframes in the data. (default is "DataSet")
            this column would contain the names of the dataframes existing in the data. e.x. 'Train' label or 'Test' label
            
        selected_features : list, optional
            list of columns to be plotted. (default is None, so function will plot all columns)
        
        save_results : bool, optional 
            if save_results is True, the function will save the plots. (default is True)
            
        groupby : str or a Pandas Series, optional 
            the column name or a Pandas Series that the function performs grouping on. (default is None)
        
        bins : list of tuples, optional 
            the intervals used by the function to cut the groupby column and perform grouping. (default is None)
            e.x. [(1, 20), (20, 50)]
            
        n_bins : int, optional 
            the number of equal-width bins used by the function to cut the groupby column and perform grouping. (default is None)
            (There is no guarantee about the distribution of data in each bin. e.x. a bin may even get empty at the end)
        
        n_quantiles : int, optional 
            the number of equal-sized (not equal-width) bins used by the function to cut the groupby column and perform grouping. (default is None)
            
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
        plots matrix of the histograms  
        """

        data = self._concat_n_datasets(data, joined_col_name)

        if 'spark.sql' in str(type(data)): #convert to pyspark.pandas df, if it is pyspark.sql df
            data = data.pandas_api()

        if len(self.feat_names) < data.shape[1] and selected_features is None:
            selected_features = self.feat_names
        if len(self.feat_names) < data.shape[1] and plot_kwargs.get('hue_col') is None:
            plot_kwargs['hue_col'] = joined_col_name

        if plot_kwargs.get('hue_col'):
            plot_kwargs['hue_order'] = data[plot_kwargs.get('hue_col')].unique()

        groups, grouping_var = self._get_groups(data, groupby, bins=bins, n_bins=n_bins, n_quantiles=n_quantiles)
        self._create_saving_folders(self._save_folder, groups, grouping_var) if save_results else 0

        for grp_name, indices in groups.items():
            group_name = f"{grouping_var}={str(grp_name)}"
            saving_path = f"{self._save_folder}/{group_name}" if save_results else None
            if 'spark' in str(type(data)):
                plot_histograms_of_data_v3(data.iloc[indices.to_numpy()], selected_features, saving_path, group_name, plot_kwargs)
            else:
                plot_histograms_of_data_v3(data.loc[indices], selected_features, saving_path, group_name, plot_kwargs)

    def plot_outliers(self, data, selected_features=None, joined_col_name="DataSet", save_results=True,
                      groupby=None, bins=None, n_bins=None, n_quantiles=None,
                      **plot_kwargs):
        """
        plots the boxplots of columns   
        
        Parameters
        ----------
        data : DataFrame object or dict
            a single dataframe object or a dictionary of some dataframes which will be concatenated. e.x. {'Train':Xy_train,'Test':Xy_test}
        joined_col_name : str, optional 
            if input data is dict, then this is the name of the new column added to the concatenated dataframes in the data. (default is "DataSet")
            this column would contain the names of the dataframes existing in the data. e.x. 'Train' label or 'Test' label
        selected_features : list, optional
            list of columns to be plotted. (default is None, so function will plot all columns)
        
        save_results : bool, optional 
            if save_results is True, the function will save the plots. (default is True)
            
        groupby : str or a Pandas Series, optional 
            the column name or a Pandas Series that the function performs grouping on. (default is None)
        
        bins : list of tuples, optional 
            the intervals used by the function to cut the groupby column and perform grouping. (default is None)
            e.x. [(1, 20), (20, 50)]
            
        n_bins : int, optional 
            the number of equal-width bins used by the function to cut the groupby column and perform grouping. (default is None)
            (There is no guarantee about the distribution of data in each bin. e.x. a bin may even get empty at the end)
        
        n_quantiles : int, optional 
            the number of equal-sized (not equal-width) bins used by the function to cut the groupby column and perform grouping. (default is None)
            
        **plot_kwargs : Additional keyword arguments, optional 
            hue_col : str, optional 
                Semantic variable (column) that is mapped to determine the color of plot elements. (default is None)
            n_cols : int, optional
                number of columns in plotting matrix of the subplots (default is 1)
            
            
        Returns
        -------
        plots matrix of the boxplots  
        """
        data = self._concat_n_datasets(data, joined_col_name)

        if 'spark.sql' in str(type(data)): #convert to pyspark.pandas df, if it is pyspark.sql df
            data = data.pandas_api()

        if len(self.feat_names) < data.shape[1] and plot_kwargs.get('hue_col') is None:
            plot_kwargs['hue_col'] = joined_col_name
        if plot_kwargs.get('hue_col', None):
            plot_kwargs['hue_order'] = data[plot_kwargs.get('hue_col')].unique()

        groups, grouping_var = self._get_groups(data, groupby, bins=bins, n_bins=n_bins, n_quantiles=n_quantiles)
        self._create_saving_folders(self._save_folder, groups, grouping_var) if save_results else 0

        for grp_name, indices in groups.items():
            group_name = f"{grouping_var}={str(grp_name)}"
            saving_path = f"{self._save_folder}/{group_name}" if save_results else None
            if 'spark' in str(type(data)):
                plot_outliers_v3(data.iloc[indices.to_numpy()], selected_features, saving_path, group_name, plot_kwargs)
            else:
                plot_outliers_v3(data.loc[indices], selected_features, saving_path, group_name, plot_kwargs)

    # def complete_data_assessment(self,
    def perform_data_analysis(self,
                              data, selected_features=None,
                              groupby=None, bins=None, n_bins=None, n_quantiles=None,
                              include_describe_data=True,
                              include_missing_analysis=False,
                              include_correlation_analysis=True,
                              correlation_methods = ('Pearson', 'phik'), top_corr_thr=0.7,
                              include_plot_hist=True, plots_hue_col=None,
                              include_plot_outliers=True,
                              ):

        """
          Performs an automatic data analysis

          Parameters
          ----------
          data : DataFrame object or dict
          a single dataframe object or a dictionary of some dataframes which will be concatenated. e.x. {'Train':Xy_train,'Test':Xy_test}
          selected_features : list, optional
          list of columns to be plotted. (default is None, so function will plot all columns)
          if report is True, the function will make a report. (default is True)
          groupby : str or a Pandas Series, optional
          the column name or a Pandas Series that the function performs grouping on. (default is None)
          bins : list of tuples, optional
          the intervals used by the function to cut the groupby column and perform grouping. (default is None)
          e.x. [(1, 20), (20, 50)]
          n_bins : int, optional
          the number of equal-width bins used by the function to cut the groupby column and perform grouping. (default is None)
          (There is no guarantee about the distribution of data in each bin. e.x. a bin may even get empty at the end)
          n_quantiles : int, optional
          the number of equal-sized (not equal-width) bins used by the function to cut the groupby column and perform grouping. (default is None)
          include_describe_data : bool, optional
          whether include data description or not. (default is True)
          include_missing_analysis : bool, optional
          whether include missing analysis or not. (default is False)
          include_correlation_analysis : bool, optional
          whether include correlation analysis or not. (default is True)
          include_plot_hist : bool, optional
          whether include distribution analysis or not. (default is True)
          include_plot_outliers : bool, optional
          whether include outlier analysis or not. (default is True)
          correlation_methods :tuple, optional
          the correlation methods to be included. default is ('Pearson', 'phik')
          top_corr_thr: float, optional
          the correlation threshold for making tables of features' correlation with values over threshold. (default is 0.7)
          plots_hue_col: str, optional
          the hue column.(default is None)
          if it gets None and input data is dict of two datasets instead of single dataframe, then the hue would be based on the name of datasets that are keys of dictionary.
          Returns
          -------
          some plots, excel files and a report.
        """

        if include_describe_data:
            self.describe_data(data, selected_features=selected_features, save_results=True,
                               groupby=groupby, bins=bins, n_bins=n_bins, n_quantiles=n_quantiles)

        if include_missing_analysis:
            self.missing_analysis(data, selected_features=selected_features, joined_col_name="DataSet",
                                  groupby=groupby, bins=bins, n_bins=n_bins, n_quantiles=n_quantiles,
                                  save_results=True)

        if include_correlation_analysis:
            self.correlation_analysis(data, joined_col_name="DataSet", methods=correlation_methods,
                                      top_corr_thr=top_corr_thr,
                                      groupby=groupby, bins=bins, n_bins=n_bins, n_quantiles=n_quantiles,
                                      save_results=True)

        if include_plot_hist:
            self.plot_hist(data, selected_features=None, joined_col_name="DataSet", save_results=True,
                           groupby=groupby, bins=bins, n_bins=n_bins, n_quantiles=n_quantiles,
                           hue_col=plots_hue_col)

        if include_plot_outliers:
            self.plot_outliers(data, selected_features=None, joined_col_name="DataSet", save_results=True,
                               groupby=groupby, bins=bins, n_bins=n_bins, n_quantiles=n_quantiles,
                               hue_col=plots_hue_col)
        
    
