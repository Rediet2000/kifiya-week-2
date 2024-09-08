from sqlalchemy import create_engine
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Create a connection using SQLAlchemy
def create_connection():
    try:
        db_url = "postgresql://postgres:new_password@localhost:5432/telecom"
        engine = create_engine(db_url)
        print("Database connection successful!")
        return engine
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

# Step 2: Fetch data from xdr_data table
def fetch_telecom_data(engine):
    try:
        query = """
        SELECT 
            "MSISDN/Number" as msisdn,
            "Start", 
            "End", 
            "Total UL (Bytes)", 
            "Total DL (Bytes)", 
            "Social Media DL (Bytes)",
            "Google DL (Bytes)", 
            "Youtube DL (Bytes)"
        FROM public.xdr_data;
        """
        data = pd.read_sql(query, engine)
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Step 3: Calculate engagement metrics
def calculate_metrics(df):
    df['Start'] = pd.to_datetime(df['Start'], errors='coerce')
    df['End'] = pd.to_datetime(df['End'], errors='coerce')
    
    # Calculate session duration
    df['session_duration'] = (df['End'] - df['Start']).dt.total_seconds()
    
    # Calculate total traffic per session (upload + download)
    df['total_traffic'] = df['Total UL (Bytes)'] + df['Total DL (Bytes)']
    
    # Group by MSISDN to calculate metrics
    user_engagement = df.groupby('msisdn').agg(
        session_frequency=('msisdn', 'count'),
        avg_session_duration=('session_duration', 'mean'),
        total_traffic=('total_traffic', 'sum')
    ).reset_index()
    
    return user_engagement

# Step 4: Normalize metrics and apply K-Means clustering
def normalize_and_cluster(df, n_clusters=3):
    scaler = MinMaxScaler()
    
    # Select columns to normalize
    metrics = ['session_frequency', 'avg_session_duration', 'total_traffic']
    
    # Normalize the metrics
    df_normalized = df.copy()
    df_normalized[metrics] = scaler.fit_transform(df[metrics])
    
    # Apply K-Means with k=3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_normalized['cluster'] = kmeans.fit_predict(df_normalized[metrics])
    
    return df_normalized, kmeans

# Step 5: Calculate cluster stats
def compute_cluster_stats(df, cluster_col='cluster'):
    cluster_stats = df.groupby(cluster_col).agg(
        min_session_frequency=('session_frequency', 'min'),
        max_session_frequency=('session_frequency', 'max'),
        avg_session_frequency=('session_frequency', 'mean'),
        total_session_frequency=('session_frequency', 'sum'),
        min_avg_session_duration=('avg_session_duration', 'min'),
        max_avg_session_duration=('avg_session_duration', 'max'),
        avg_avg_session_duration=('avg_session_duration', 'mean'),
        total_avg_session_duration=('avg_session_duration', 'sum'),
        min_total_traffic=('total_traffic', 'min'),
        max_total_traffic=('total_traffic', 'max'),
        avg_total_traffic=('total_traffic', 'mean'),
        total_total_traffic=('total_traffic', 'sum')
    ).reset_index()
    
    return cluster_stats

# Step 6: Application-based analysis
def application_based_traffic(df):
    app_cols = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Youtube DL (Bytes)']
    
    # Sum traffic per application and msisdn
    app_traffic = df.groupby('msisdn')[app_cols].sum().reset_index()
    
    # Find top 10 users by application usage
    app_traffic['total_app_traffic'] = app_traffic[app_cols].sum(axis=1)
    top_10_users = app_traffic.nlargest(10, 'total_app_traffic')
    
    return top_10_users

# Step 7: Plotting top 3 used applications
def plot_top_3_apps(df):
    app_cols = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Youtube DL (Bytes)']
    app_traffic = df[app_cols].sum()
    
    # Plot the top 3 apps
    plt.figure(figsize=(8, 6))
    sns.barplot(x=app_traffic.index, y=app_traffic.values)
    plt.title("Top 3 Most Used Applications")
    plt.ylabel("Total Traffic (Bytes)")
    plt.show()

# Step 8: Elbow method for optimal K
def elbow_method(df):
    metrics = ['session_frequency', 'avg_session_duration', 'total_traffic']
    distortions = []
    K = range(1, 10)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df[metrics])
        distortions.append(kmeans.inertia_)
    
    # Plot elbow curve
    plt.figure(figsize=(8, 6))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    plt.show()

# Step 9: Main function to run the engagement analysis
def main():
    engine = create_connection()
    if engine is None:
        return
    
    telecom_data = fetch_telecom_data(engine)
    if telecom_data is None:
        return
    
    # Calculate engagement metrics
    engagement_metrics = calculate_metrics(telecom_data)
    
    # Normalize and apply k-means clustering
    df_normalized, kmeans = normalize_and_cluster(engagement_metrics, n_clusters=3)
    
    # Compute cluster statistics
    cluster_stats = compute_cluster_stats(df_normalized)
    print("Cluster Stats:\n", cluster_stats)
    
    # Top 10 users by application-based traffic
    top_10_users = application_based_traffic(telecom_data)
    print("Top 10 Users by Application-based Traffic:\n", top_10_users)
    
    # Plot top 3 most used applications
    plot_top_3_apps(telecom_data)
    
    # Determine optimal k using the elbow method
    elbow_method(engagement_metrics)

if __name__ == "__main__":
    main()
