import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
from DataProfiling_utils import *
from DataProfiler import DataProfiler  # Ensure your DataProfiler class is in DataProfiler.py
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import time
import zipfile
from io import BytesIO


# Configure the page
st.set_page_config(
    page_title="Data Profiler Dashboard",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

# App title
st.title("📊 Data Profiler Dashboard")

# User Guide Expander
with st.expander("User Guide"):
    st.markdown("""
    **Welcome to the Data Profiler Dashboard!**

    This application allows you to perform comprehensive data profiling on your datasets with ease. Data profiling is a crucial step in data analysis and preprocessing, helping you understand the structure, content, and quality of your data. It is widely used in industries for data cleaning, validation, and exploratory data analysis.

    **How to Use:**

    - **Select Dataset:** Use the sidebar to upload your own CSV dataset or load the example breast cancer dataset.
    - **Select Features:** Choose the features (columns) you wish to include in the analysis.
    - **Binning Options:** Select a binning method for histograms (Equal-width Bins or Quantile Bins) and specify the number of bins.
    - **Analysis Options:** Choose the types of analyses you want to perform:
        - **Descriptive Statistics:** Provides statistical summaries for each column.
        - **Missing Value Analysis:** Shows the count and percentage of missing values in each column.
        - **Correlation Analysis:** Computes correlation matrices using selected methods and highlights correlation values greater than a specified threshold.
        - **Histograms:** Plots histogram plots for data columns.
        - **Box Plots:** Generates box plots to visualize data distribution and identify outliers.
    - **Correlation Methods:** Select correlation methods (Pearson, Spearman, Kendall) and set a correlation threshold to highlight significant correlations. Please note this option is computationally intensive and may take some time.
    - **Run Profiling:** Click the **Run Profiling** button to start the analysis.

    **Example Dataset:**
    If you don't want to upload your own data and just want to see how the app works, you can load the California Housing dataset from the sidebar.
    This dataset includes various features about housing in California, such as median income, housing age, total rooms, and more.
    It's an excellent example to test the app's capabilities and explore the data profiling features.
                
    **Data Profiling in Industry:**

    Data profiling is essential in industries such as finance, healthcare, retail, and more. It helps in:

    - Identifying data quality issues.
    - Understanding data distributions.
    - Detecting anomalies and outliers.
    - Informing data cleaning and preprocessing steps.
    - Supporting data-driven decision-making.

    **Enjoy exploring your data!**
    """)

# Sidebar for user inputs
st.sidebar.header("Dataset Selection")
dataset_option = st.sidebar.radio("Select Dataset", ("Upload Your Own Data", "California Housing Dataset as an Example"))

if dataset_option == "Upload Your Own Data":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        # Read the uploaded CSV file
        try:
            st.session_state['data'] = pd.read_csv(uploaded_file)
            st.success("Dataset loaded successfully!")
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.stop()
    else:
        st.info("Awaiting CSV file upload.")
        st.stop()
elif dataset_option == "California Housing Dataset as an Example":
    # Load the breast cancer dataset
    st.session_state['data'] = pd.read_csv(r'''Examples\Data\HousingData_TrainData.csv''')
    st.success("Example dataset loaded successfully!")

# Display the dataset
if st.checkbox("Show raw data"):
    st.subheader("Raw Data")
    st.dataframe(st.session_state.get('data',pd.DataFrame()).head(10))

# Initialize DataProfiler
temp_dir = tempfile.TemporaryDirectory()
profiler = DataProfiler(save_folder_name=temp_dir.name)  # Save results to temporary directory

# Feature selection
st.sidebar.header("Feature Selection")
if st.session_state.get('data', None) is not None:
    selected_features = st.sidebar.multiselect(
        "Select features to include in the analysis", st.session_state.get('data', pd.DataFrame()).columns.tolist(), default= st.session_state.get('data', pd.DataFrame()).columns.tolist()
    )

# Binning options
st.sidebar.header("Binning Options")
binning_method = st.sidebar.selectbox(
    "Select binning method", ["Equal-width Bins", "Quantile Bins"]
)

# Initialize session state variables for bins
if 'n_bins' not in st.session_state:
    st.session_state['n_bins'] = None
if 'n_quantiles' not in st.session_state:
    st.session_state['n_quantiles'] = None

if binning_method == "Equal-width Bins":
    st.session_state['n_quantiles'] = None
    n_bins_input = st.sidebar.text_input("Number of equal-width bins (integer)", value="5")
    try:
        st.session_state['n_bins'] = int(n_bins_input)
    except ValueError:
        st.error("Please enter a valid integer for the number of equal-width bins.")
        st.session_state['n_bins'] = None
elif binning_method == "Quantile Bins":
    st.session_state['n_bins'] = None
    n_quantiles_input = st.sidebar.text_input("Number of quantile bins (integer)", value="5")
    try:
        st.session_state['n_quantiles'] = int(n_quantiles_input)
    except ValueError:
        st.error("Please enter a valid integer for the number of quantile bins.")
        st.session_state['n_quantiles'] = None

# Analysis options
st.sidebar.header("Analysis Options")
include_describe_data = st.sidebar.checkbox("Include Descriptive Statistics", value=True)
include_missing_analysis = st.sidebar.checkbox("Include Missing Value Analysis", value=True)
include_correlation_analysis = st.sidebar.checkbox("Include Correlation Analysis", value=True)
include_plot_hist = st.sidebar.checkbox("Include Histograms", value=True)
include_plot_outliers = st.sidebar.checkbox("Include Box Plots", value=True)

# Correlation methods and threshold
correlation_methods = st.sidebar.multiselect(
    "Correlation Methods", ["Pearson", "Spearman", 'Kendall'], default=["Kendall"]
)
st.session_state['top_corr_thr'] = st.sidebar.text_input("Correlation Threshold", value=0.7)

# Run the profiler
if st.sidebar.button("Run Profiling"):
    with st.spinner("Running data profiling..."):
        
        ##### Perform data analysis using the dataprofiler.py module
        # # Progress indicator
        # progress_bar = st.progress(0)
        # progress_text = st.empty()
        # progress = 0

        # # Perform data analysis using the dataprofiler.py module
        # profiler.perform_data_analysis(
        #     st.session_state['data'][selected_features],
        #     selected_features=selected_features,
        #     groupby=None,  # Removed groupby feature from dashboard
        #     bins=None,  # Removed custom binning feature from dashboard
        #     n_bins=st.session_state.get('n_bins', None),
        #     n_quantiles=st.session_state.get('n_quantiles', None),
        #     include_describe_data=include_describe_data,
        #     include_missing_analysis=include_missing_analysis,
        #     include_correlation_analysis=include_correlation_analysis,
        #     correlation_methods=tuple([method.lower() for method in correlation_methods]),
        #     top_corr_thr=st.session_state.get('top_corr_thr', None),
        #     include_plot_hist=include_plot_hist,
        #     include_plot_outliers=include_plot_outliers,
        # )

        # progress += 50
        # progress_bar.progress(progress)
        # progress_text.text("Generating report...")

        # # Simulate progress
        # time.sleep(1)
        # progress += 50
        # progress_bar.progress(progress)
        # progress_text.text("Analysis complete!")

        # st.success("Data profiling completed!")

        # Display results
        st.divider()

        st.header("Analysis Results")
        st.divider()

        # Descriptive statistics
        if include_describe_data:
            st.subheader("Descriptive Statistics")
            describe_result = describe_data(st.session_state['data'][selected_features])
            st.dataframe(describe_result)
        
        st.divider()

        # Missing value analysis
        if include_missing_analysis:
            st.subheader("Missing Value Analysis")
            missing_result = missing_count(st.session_state['data'][selected_features])
            st.dataframe(missing_result)
        
        st.divider()

        # Correlation analysis
        if include_correlation_analysis:
            st.subheader("Correlation Analysis")
            for method in correlation_methods:

                # Get the list of numeric columns from the data
                numeric_cols = st.session_state['data'].select_dtypes(include=[np.number]).columns.tolist()
                # exclude  non-numeric columns in selected_features, some categorical columns are string and therefore not suitable for correlation analysis
                numeric_selected_features = [col for col in selected_features if col in numeric_cols]
                
                corr_result = st.session_state['data'][numeric_selected_features].corr(method=method.lower())
                st.write(f"**{method} Correlation Matrix**")

                # Define a function to highlight cells based on the threshold
                def highlight_corr(val):
                    color = 'yellow' if abs(val) > float(st.session_state.get('top_corr_thr', 0.7)) else ''
                    return f'background-color: {color}'

                # Apply the styling to the correlation matrix
                styled_corr = corr_result.style.applymap(highlight_corr).format("{:.3f}")
                st.dataframe(styled_corr)

                # Display heatmap using Plotly
                fig = px.imshow(
                    corr_result,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    origin='upper',
                )
                fig.update_layout(
                    title=f"{method} Correlation Heatmap",
                    autosize=False,
                    width=600,
                    height=600,
                )
                st.plotly_chart(fig)

        st.divider()
        # Histograms

        # Define colors
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

        if include_plot_hist:
            st.subheader("Histograms")
            for idx, feat in enumerate(selected_features):
                color = colors[idx % len(colors)]  ### Cycle through colors for features

                if st.session_state['data'][selected_features][feat].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    if binning_method == "Equal-width Bins" and st.session_state.get('n_bins', None) is not None:
                        ###mthod 1 # Use the specified number of bins #somehow this is not working accuartely and shows less number of bins
                        # fig = px.histogram(st.session_state['data'][selected_features], x=feat, nbins=st.session_state.get('n_bins', None))

                        ###method 2
                        # Use the specified number of bins by setting xbins
                        data_min = st.session_state['data'][selected_features][feat].min()
                        data_max = st.session_state['data'][selected_features][feat].max()
                        n_bins = st.session_state.get('n_bins', 10)  # Default to 10 bins if not specified

                        # Slightly adjust the data range to include all data points
                        data_range = data_max - data_min
                        bin_size = data_range / n_bins
                        data_min_adjusted = data_min - (data_range * 0.001)  # Subtract 0.1% of the range
                        data_max_adjusted = data_max + (data_range * 0.001)  # Add 0.1% of the range

                        fig = px.histogram(st.session_state['data'][selected_features], x=feat)

                        fig.update_traces(
                            xbins=dict(
                                start=data_min_adjusted,
                                end=data_max_adjusted,
                                size=bin_size
                            ),
                            marker_color=color)

                        fig.update_layout(
                            xaxis_title=feat,
                            yaxis_title='Count',
                            bargap=0.1
                        )
                    elif binning_method == "Quantile Bins" and st.session_state.get('n_quantiles', None) is not None:
                        # Bin data into quantiles
                        try:
                            binned_data = st.session_state['data'][selected_features].copy()
                            n_quantiles = st.session_state.get('n_quantiles', None)
                            unique_values = binned_data[feat].dropna().unique()
                            n_unique = len(unique_values)

                            if n_unique == 2:
                                # Handle binary columns
                                def map_binary_to_bins(x):
                                    if x == 0:
                                        return f'Bin 1'
                                    elif x == 1:
                                        return f'Bin {n_quantiles}'
                                    else:
                                        return f'Bin {n_quantiles // 2}'

                                binned_data['binned'] = binned_data[feat].map(map_binary_to_bins)
                                category_order = {'binned': [f'Bin {i}' for i in range(1, n_quantiles + 1)]}
                            else:
                                binned_data['binned'] = pd.qcut(
                                    binned_data[feat], q=n_quantiles, duplicates='drop'
                                ).astype(str)
                                category_order = None
                            fig = px.histogram(
                                binned_data, x='binned', color_discrete_sequence=[color], category_orders=category_order
                            )
                            fig.update_xaxes(title=f"{feat} (Quantile Bins)")
                            fig.update_yaxes(title="Count")
                            fig.update_layout(showlegend=False)
                        except Exception as e:
                            st.error(f"Error binning data for feature {feat}: {e}")
                            continue
                    else:
                        # No binning or invalid binning parameters
                        fig = px.histogram(st.session_state['data'][selected_features], x=feat)
                    st.plotly_chart(fig)

        st.divider()

        # Box plots
        if include_plot_outliers:
            st.subheader("Box Plots")
            for idx, feat in enumerate(selected_features):
                color = colors[idx % len(colors)]
                if st.session_state['data'][selected_features][feat].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    fig = px.box(st.session_state['data'][selected_features], x=feat, color_discrete_sequence=[color])
                    st.plotly_chart(fig)

        # ## Provide download link for the generated report
        # st.header("Download Report")
        # # Zip the contents of the temporary directory
        # zip_buffer = BytesIO()
        # with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        #     for root, dirs, files in os.walk(temp_dir.name):
        #         for file in files:
        #                 file_path = os.path.join(root, file)
        #                 zip_file.write(file_path, arcname=os.path.relpath(file_path, temp_dir.name))
        # zip_buffer.seek(0)

        # st.download_button(
        #     label="Download Report as ZIP",
        #     data=zip_buffer,
        #     file_name="data_profiling_report.zip",
        #     mime="application/zip",
        # )

        st.success("Analysis results are displayed above.")
