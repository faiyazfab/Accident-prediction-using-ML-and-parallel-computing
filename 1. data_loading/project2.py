#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import dask.dataframe as dd
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import dask_xgboost as dxgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def preprocess_function(partition):
    columns_to_drop = [
        'Description', 'End_Lat', 'End_Lng', 'Zipcode', 'Weather_Timestamp',
        'Airport_Code', 'Amenity', 'Astronomical_Twilight', 'Bump', 'City',
        'Civil_Twilight', 'Country', 'County', 'Crossing', 'Give_Way', 'Humidity(%)',
        'Junction', 'Nautical_Twilight', 'No_Exit', 'Precipitation(in)', 'Pressure(in)',
        'Railway', 'Roundabout', 'Source', 'State', 'Station', 'Stop', 'Street',
        'Sunrise_Sunset', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop',
        'Wind_Chill(F)', 'Wind_Direction'
    ]
    partition = partition.drop(columns=columns_to_drop, errors="ignore")
    partition['Temperature(F)'] = partition['Temperature(F)'].fillna(partition['Temperature(F)'].mean())
    partition['Wind_Speed(mph)'] = partition['Wind_Speed(mph)'].fillna(partition['Wind_Speed(mph)'].mean())
    partition['Visibility(mi)'] = partition['Visibility(mi)'].fillna(partition['Visibility(mi)'].mean())
    partition['Weather_Condition'] = partition['Weather_Condition'].fillna('Unknown')
    partition['Timezone'] = partition['Timezone'].fillna('Unknown')
    partition = partition.dropna(subset=['Severity', 'Start_Time'])

    partition['Start_Time'] = pd.to_datetime(partition['Start_Time'], errors='coerce')
    partition['End_Time'] = pd.to_datetime(partition['End_Time'], errors='coerce')
    partition = partition.dropna(subset=['Start_Time', 'End_Time'])

    partition['Year'] = partition['Start_Time'].dt.year
    partition['Month'] = partition['Start_Time'].dt.month
    partition['Day'] = partition['Start_Time'].dt.day
    partition['Hour'] = partition['Start_Time'].dt.hour
    partition['Weekday'] = partition['Start_Time'].dt.weekday
    partition['Is_Rainy'] = partition['Weather_Condition'].str.contains('Rain', case=False, na=False)
    partition['Is_Snowy'] = partition['Weather_Condition'].str.contains('Snow', case=False, na=False)

    partition['High_Severity'] = (partition['Severity'] >= 3).astype(int)
    return partition

if __name__ == "__main__":
    cluster = SLURMCluster(
        queue='courses',
        cores=8,
        memory='24GB',
        walltime='04:00:00',
        job_extra=['--output=/home/wei.shao/Project/dask_%j.log']
    )
    cluster.scale(jobs=1)
    client = Client(cluster)
    print("Dask client connected with SLURM cluster successfully!")

    try:
        dataset_path = "/home/wei.shao/Project/US_Accidents_March23.csv"
        df = dd.read_csv(dataset_path, blocksize="64MB")
        print(f"Dataset loaded successfully from: {dataset_path}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

    meta = df.map_partitions(preprocess_function).compute().dtypes.to_dict()
    df_preprocessed = df.map_partitions(preprocess_function, meta=meta).persist()
    print("Data preprocessing completed successfully.")

    output_path = "/home/wei.shao/Project/Cleaned_US_Accidents_March23.csv"
    df_preprocessed.compute().to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")

    severity_counts = df_preprocessed['Severity'].value_counts().compute()
    plt.figure(figsize=(8, 5))
    severity_counts.plot(kind='bar', title='Severity Distribution', color='skyblue')
    plt.xlabel('Severity Level')
    plt.ylabel('Frequency')
    plt.savefig('/home/wei.shao/Project/Severity_Distribution.png')
    plt.close()

    hourly_severity = df_preprocessed.groupby('Hour')['Severity'].mean().compute()
    plt.figure(figsize=(10, 6))
    hourly_severity.plot(kind='line', title='Severity by Hour', marker='o', color='green')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Severity')
    plt.grid(True)
    plt.savefig('/home/wei.shao/Project/Severity_by_Hour.png')
    plt.close()

    weather_severity = df_preprocessed.groupby('Weather_Condition')['Severity'].mean().compute()
    top_weather = weather_severity.sort_values(ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    top_weather.plot(kind='bar', title='Top 10 Weather Conditions by Severity', color='orange')
    plt.xlabel('Weather Condition')
    plt.ylabel('Average Severity')
    plt.savefig('/home/wei.shao/Project/Top_Weather_Conditions_by_Severity.png')
    plt.close()

    numerical_features = ['Severity', 'Temperature(F)', 'Visibility(mi)', 'Wind_Speed(mph)']
    correlation = df_preprocessed[numerical_features].corr(method='pearson').compute()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig('/home/wei.shao/Project/Correlation_Heatmap.png')
    plt.close()

    print("Correlation with Severity:")
    print(correlation['Severity'])

    selected_features = ['Hour', 'Temperature(F)', 'Visibility(mi)', 'Is_Rainy', 'Is_Snowy']
    X = df_preprocessed[selected_features].compute()
    y = df_preprocessed['Severity'].compute()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dtrain = dxgb.DaskDMatrix(client, X_train, y_train)
    dtest = dxgb.DaskDMatrix(client, X_test, y_test)

    params = {
        'objective': 'multi:softmax',
        'num_class': 5,
        'max_depth': 6,
        'eta': 0.3
    }

    model = dxgb.train(client, params, dtrain, num_boost_round=100)
    predictions = dxgb.predict(client, model, dtest)

    accuracy = accuracy_score(y_test.compute(), predictions.compute())
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test.compute(), predictions.compute()))

    xgb.plot_importance(model)
    plt.title("Feature Importance")
    plt.savefig('/home/wei.shao/Project/Feature_Importance.png')
    plt.close()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




