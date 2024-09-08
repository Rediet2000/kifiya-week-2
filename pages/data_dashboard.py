import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import io

st.title("Exploratory Data Analysis Dashboard")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Convert 'Start' and 'End' columns to datetime if they exist
    if 'Start' in df.columns:
        df['Start'] = pd.to_datetime(df['Start'], errors='coerce')
    if 'End' in df.columns:
        df['End'] = pd.to_datetime(df['End'], errors='coerce')

    # Display the first few rows of the dataframe
    st.subheader("First Few Rows of the Data")
    st.write(df.head())

    # Describe relevant variables and data types
    st.subheader("Data Types and Description")
    st.write(df.dtypes)
    st.write(df.describe(include='all'))

    # Identifying and displaying missing data
    st.subheader("Missing Data Summary")
    missing_data = df.isnull().sum()
    st.write(missing_data[missing_data > 0])

    # Handling missing values
    # Fill missing values for numeric columns with mean, and for categorical with mode
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    # Identifying and treating outliers
    for column in numeric_cols:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[column] >= (Q1 - 1.5 * IQR)) & (df[column] <= (Q3 + 1.5 * IQR))]

    # Segment users into decile classes based on total duration if column exists
    if 'Total DL (Bytes)' in df.columns:
        df['decile_class'] = pd.qcut(df['Total DL (Bytes)'], 10, labels=False) + 1
        decile_summary = df.groupby('decile_class').agg({'Total UL (Bytes)': 'sum'})
        st.subheader("Decile Summary")
        st.write(decile_summary)

    # Basic metrics
    st.subheader("Basic Metrics")
    st.write(df.describe())

    # Non-Graphical Univariate Analysis
    st.subheader("Dispersion Parameters")
    dispersion = pd.DataFrame({
        'Variance': [df[col].var() for col in numeric_cols],
        'Standard Deviation': [df[col].std() for col in numeric_cols]
    }, index=numeric_cols)
    st.write(dispersion)

    # Graphical Univariate Analysis
    st.subheader("Graphical Univariate Analysis")
    for column in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax)
        ax.set_title(f'Distribution of {column}')
        st.pyplot(fig)
        plt.close(fig)  # Close the figure after rendering to avoid memory overload

    # Bivariate Analysis
    st.subheader("Bivariate Analysis")
    bivariate_results = {}
    apps = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    if 'Total DL (Bytes)' in df.columns:
        for app in apps:
            if app in df.columns:
                fig, ax = plt.subplots()
                sns.scatterplot(x=df[app], y=df['Total DL (Bytes)'], ax=ax)
                ax.set_title(f'Relationship between {app} and Total DL')
                ax.set_xlabel(app)
                ax.set_ylabel('Total DL (Bytes)')
                st.pyplot(fig)
                plt.close(fig)  # Close the figure after rendering to avoid memory overload
                correlation = df[[app, 'Total DL (Bytes)']].corr().iloc[0, 1]
                bivariate_results[app] = correlation
        st.write("Bivariate Analysis Results:")
        st.write(bivariate_results)

    # Correlation Matrix
    st.subheader("Correlation Matrix")
    feature_cols = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
    if all(col in df.columns for col in feature_cols):
        correlation_matrix = df[feature_cols].corr()
        st.write(correlation_matrix)

    # Dimensionality Reduction - PCA
    st.subheader("Dimensionality Reduction - PCA")
    if all(col in df.columns for col in feature_cols):
        x = df[feature_cols]
        x = StandardScaler().fit_transform(x)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(x)
        df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        explained_variance = pca.explained_variance_ratio_

        st.write("PCA Explained Variance Ratio:")
        st.write(explained_variance)

        st.write("\nPCA Interpretation:")
        st.write(f"1. Principal Component 1 explains {explained_variance[0] * 100:.2f}% of the variance.")
        st.write(f"2. Principal Component 2 explains {explained_variance[1] * 100:.2f}% of the variance.")
        st.write(f"3. Together, the first two components explain {sum(explained_variance) * 100:.2f}% of the total variance.")
        st.write(f"4. The components can be used for further analysis or visualization with reduced dimensionality.")

        # Save PCA results to CSV
        st.subheader("Download PCA Results")
        pca_csv = df_pca.to_csv(index=False)
        st.download_button(label="Download PCA results", data=pca_csv, file_name='pca_results.csv', mime='text/csv')
