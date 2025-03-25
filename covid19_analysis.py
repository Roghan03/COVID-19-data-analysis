import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load COVID-19 datasets
def load_data():
    url = 'synthetic_covid19_data.csv'  # Replace this with your local CSV file if using locally
    data = pd.read_csv(url)
    return data

# Data Preprocessing and cleaning
def preprocess_data(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['total_cases', 'total_deaths', 'total_vaccinations'], inplace=True)
    return df

# Correlation analysis
def correlation_analysis(df):
    df_corr = df[['total_cases', 'total_deaths', 'total_vaccinations', 'total_population', 'gdp']].dropna()
    correlation_matrix = df_corr.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of COVID-19 Data")
    plt.show()

# Trend Analysis using Moving Averages
def plot_moving_average_trends(df, country='World', window=7):
    country_data = df[df['location'] == country]
    country_data['moving_avg_cases'] = country_data['total_cases'].rolling(window=window).mean()
    country_data['moving_avg_deaths'] = country_data['total_deaths'].rolling(window=window).mean()
    country_data['moving_avg_vaccinations'] = country_data['total_vaccinations'].rolling(window=window).mean()
    
    plt.figure(figsize=(12,6))
    plt.plot(country_data['date'], country_data['moving_avg_cases'], label='7-Day Moving Average (Cases)', color='blue')
    plt.plot(country_data['date'], country_data['moving_avg_deaths'], label='7-Day Moving Average (Deaths)', color='red')
    plt.plot(country_data['date'], country_data['moving_avg_vaccinations'], label='7-Day Moving Average (Vaccinations)', color='green')
    plt.title(f"COVID-19 Moving Average Trends Over Time for {country}")
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

# Predict COVID-19 cases using Linear Regression
def predict_covid_cases(df, country='United States'):
    country_data = df[df['location'] == country]
    country_data['date'] = pd.to_datetime(country_data['date'])
    country_data['date_ordinal'] = country_data['date'].apply(lambda x: x.toordinal())  # Convert date to ordinal
    X = country_data[['date_ordinal']]
    y = country_data['total_cases']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae}")
    
    plt.figure(figsize=(12,6))
    plt.plot(country_data['date'], country_data['total_cases'], label='Actual Cases', color='blue')
    plt.plot(country_data['date'].iloc[len(X_train):], y_pred, label='Predicted Cases', color='red')
    plt.title(f"COVID-19 Cases Prediction for {country}")
    plt.xlabel('Date')
    plt.ylabel('Total Cases')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

# K-Means Clustering of countries based on infection/vaccination rates
def cluster_countries(df):
    country_data = df[['location', 'total_cases', 'total_vaccinations', 'total_deaths', 'total_population']].dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(country_data[['total_cases', 'total_vaccinations', 'total_deaths']])
    kmeans = KMeans(n_clusters=3, random_state=42)
    country_data['Cluster'] = kmeans.fit_predict(scaled_data)
    
    plt.figure(figsize=(10,6))
    sns.scatterplot(x='total_cases', y='total_vaccinations', hue='Cluster', data=country_data, palette='Set1')
    plt.title("Clustering of Countries Based on COVID-19 Data")
    plt.xlabel('Total Cases')
    plt.ylabel('Total Vaccinations')
    plt.show()
    
    return country_data[['location', 'Cluster']]

# Main function to run the analysis
def main():
    df = load_data()  # Load dataset
    df = preprocess_data(df)  # Preprocess the data
    
    # Perform Analysis
    correlation_analysis(df)  # Perform correlation analysis
    plot_moving_average_trends(df, country='United States')  # Plot trends for a specific country
    predict_covid_cases(df, country='United States')  # Predict COVID-19 cases for a country
    clustered_data = cluster_countries(df)  # Perform K-Means clustering
    print(clustered_data.head())  # Display clustering results

if __name__ == "__main__":
    main()
