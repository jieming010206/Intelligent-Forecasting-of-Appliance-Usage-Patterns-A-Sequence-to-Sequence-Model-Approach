import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


from sklearn.cluster import KMeans
import pandas as pd

def convert_discrete(data, k_means=True):
    data_binary = pd.DataFrame(index=data.index)
    data_binary['timestamp'] = data['timestamp']

    if k_means:
        for column in data.columns[2:]:
            data_column = data[column].values.reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
            kmeans.fit(data_column)
            if kmeans.cluster_centers_[0][0] > kmeans.cluster_centers_[1][0]:
                kmeans.labels_ = 1 - kmeans.labels_
            data_binary[column] = kmeans.labels_
    else:
        for column in data.columns[2:]:
            data_binary[column] = (data[column] > 0).astype(int)
            
    return data_binary



def extract_info(df):

    cols = df.columns.tolist()
    new_cols = [cols[0]] + list(range(len(cols) - 1))
    df.columns = new_cols


    df_melted = df.melt(id_vars='timestamp', var_name='appliance', value_name='status')

    # Step 2: Sort values and create additional column to indicate whether the appliance status has changed from the last minute
    df_melted.sort_values(['appliance', 'timestamp'], inplace=True)
    df_melted['change'] = df_melted.groupby('appliance')['status'].diff().ne(0).astype(int)

    # Step 3: Calculate cumulative sum over the 'change' column within each appliance and this forms the group of continuous periods of the same appliance status
    df_melted['period'] = df_melted.groupby('appliance')['change'].cumsum()

    # Step 4: Filter out only activated appliance records
    df_activated = df_melted[df_melted['status'] == 1]

    # Step 5: Create a new dataframe, calculating duration and starting time of each continuous period of activated appliance
    df_final = df_activated.groupby(['appliance', 'period']).agg(activated_time=('timestamp', 'min'),
                                                                  duration=('timestamp', 'count')).reset_index()

    # Step 6: Drop the 'period' column and sort by 'activated_time'
    df_final.drop(columns='period', inplace=True)
    df_final.sort_values(by='activated_time', inplace=True)

    # Convert 'activated_time' into your desired integer format
    df_final["duration"] = df_final["duration"].apply(lambda x: x - 1 if x < 7 else 6)
    df_final["timestamp"] = pd.to_datetime(df_final["activated_time"])
    df_final["week"] = df_final["timestamp"].dt.week
    df_final["day"] = df_final["timestamp"].dt.dayofyear
    df_final["day_week"] = df_final["timestamp"].dt.dayofweek
    df_final["activated_time"] = df_final['timestamp'].dt.hour 
    df_final = df_final.drop("timestamp", axis = 1)
    

    df_final = df_final.reset_index(drop = True)
    df_final = df_final[["week", "day", "day_week", "activated_time", "appliance", "duration"]]

    return df_final

def extract_event_series(df):
    # Rename columns for the melt function
    cols = df.columns.tolist()
    new_cols = [cols[0]] + list(range(len(cols) - 1))
    df.columns = new_cols
    
    # Melt the dataframe
    df_melted = df.melt(id_vars='timestamp', var_name='appliance', value_name='status')
    

    # Filter out only activated appliance records
    df_activated = df_melted[df_melted['status'] == 1]
    
    # Sort values by appliance and timestamp
    df_activated.sort_values(['timestamp', 'appliance'], inplace=True)
    
    # Convert 'timestamp' to datetime format
    df_activated["timestamp"] = pd.to_datetime(df_activated["timestamp"])
    
    # Extract desired time features
    df_activated["week"] = df_activated["timestamp"].dt.week
    df_activated["year"] = df_activated["timestamp"].dt.year - 2022
    df_activated["day"] = df_activated["timestamp"].dt.dayofyear
    df_activated["month"] = df_activated["timestamp"].dt.month - 1
    df_activated["day_month"] = df_activated["timestamp"].dt.day - 1
    df_activated["day_week"] = df_activated["timestamp"].dt.dayofweek
    df_activated["activated_time"] = df_activated['timestamp'].dt.hour * 6 + df_activated['timestamp'].dt.minute // 10
    
    # Select only the required columns
    df_final = df_activated[["week", "day", "year", "month", "day_month", "day_week", "activated_time", "appliance"]].reset_index(drop=True)
    
    return df_final

if __name__ == "__main__":
    data = pd.read_csv(
        "data/bq-results-20230811-124310-1691757929862.csv",
        header=0,
        names=[
            "house_id",
            "timestamp",
            "Heat Pump",
            "Tumble-Dryer",
            "Microwave",
            "TV",
            "Oven",
            "Clothes Washer",
            "Toaster",
            "Dish Washer",
            "Air Conditioner",
            "Coffee Maker",
            "Electric Kettle",
        ],
    )

    data.rename(
        columns={
            "Heat Pump": 0,
            "Tumble-Dryer": 1,
            "Microwave": 2,
            "TV": 3,
            "Oven": 4,
            "Clothes Washer": 5,
            "Toaster": 6,
            "Dish Washer": 7,
            "Air Conditioner": 8,
            "Coffee Maker": 9,
            "Electric Kettle": 10,
        },
        inplace=True,
    )

    data = data.fillna(0)
    house_ids = data["house_id"].unique()
    appliances = data.columns[2:]
    print(f"number of house: {len(house_ids):f}")

    j = 0
    num = 0
    ids = []
    for i in range(len(house_ids)):
        
        data_id_i = data[data["house_id"] == house_ids[i]]
        if len(data_id_i) > 52560:
            ids.append(house_ids[i])
            data_id_i = convert_discrete(data_id_i)
            transformed_data_id_i = extract_event_series(data_id_i)
            transformed_data_id_i.to_csv(f"./data/house_ids_{j}.csv", index=False)
            j += 1
        else:
            continue
    print(ids)


