import pandas as pd
from glob import glob
from pathlib import Path

folder_path = "starter/stat_data_folder"
glob_paths = glob(f"{folder_path}/*.csv")
grouped_tuple_list = []
df_list = []
for file_path in glob_paths:
    date = (Path(file_path).stem).split('_')[-1]
    # Load the original CSV data
    original_df = pd.read_csv(file_path)

    # Initialize an empty list to store the transformed data
    transformed_data = []

    # Iterate through the rows of the original DataFrame
    for index, row in original_df.iterrows():
        ticker = row['Ticker']
        for bin_num in range(1, 79):  # Assuming there are 78 bins
            bin_name = f'bin{bin_num}'
            metric_value = float(row[bin_name])
            if len(ticker) <= 3:
                ticker_venue = "NYSE"
            else:
                ticker_venue = "NASDAQ"
            transformed_data.append([ticker_venue, bin_name, metric_value])

    # Create a new DataFrame from the transformed data
    transformed_df = pd.DataFrame(transformed_data, columns=["Venue", 'Bin', 'Value'])

    grouped = transformed_df.groupby(['Venue', 'Bin'])['Value']

    stats_df = pd.DataFrame({
        'mean': grouped.mean(),
        'stdv': grouped.std(),
        'p50': grouped.median(),
        'p2.5': grouped.quantile(0.025),
        'p97.5': grouped.quantile(0.975)
    }).reset_index()
    
    stats_df.sort_values(by=['Bin', 'Venue'])
    
    stats_df['date'] = date
    
    df_list.append(stats_df)

result = pd.concat(df_list).reset_index()
result.to_csv('starter/stat_data_folder/stat_computed.csv')


