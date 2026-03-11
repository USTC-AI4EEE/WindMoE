import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

ROOT_PATH = '/home/wangqi/code/windmoe/datasets/goldwind/typhoon'
OUTPUT_PATH = '/home/wangqi/code/windmoe/datasets/goldwind/datacsv'

STATION_CAP_MAP = {
    1700: 40000,
    402: 208000
}

STATION_ID = 402
NWP_FEATURES = ['p_sfc', 'rh_2', 't_2', 'tdew2m', 'wdir_70', 'wspd_70']
POWER_FEATURE = 'power'
ALL_FEATURES = NWP_FEATURES + [POWER_FEATURE]

def save_csv(data_df: pd.DataFrame, suffix: str, station: int, columns_to_save: list):
    
    if data_df.empty:
        print(f"Data for '{suffix}' is empty. Skipping save.")
        return
    valid_columns = [col for col in columns_to_save if col in data_df.columns]
    output_file = os.path.join(OUTPUT_PATH, f'goldwind_typhoon_{station}_{suffix}.csv')
    data_df[valid_columns].to_csv(output_file, index=True)
    print(f"Saved {suffix} data with shape {data_df[valid_columns].shape} to: {output_file}")


def main(station_id):

    print(f"--- Starting Typhoon Data Preprocessing for station {station_id} ---")

    measured_path = os.path.join(ROOT_PATH, f'measured/{station_id}.csv')
    predicted_path = os.path.join(ROOT_PATH, f'predicted/{station_id}_forecast.csv')
    
    measured_df = pd.read_csv(measured_path, encoding='gbk')
    predicted_df = pd.read_csv(predicted_path, encoding='gbk')

    measured_df['dtime'] = pd.to_datetime(measured_df['dtime'], format='%d/%m/%Y %H:%M')
    predicted_df['dtime'] = pd.to_datetime(predicted_df['dtime'], format='%Y-%m-%d %H:%M:%S')
    
    df = pd.merge(measured_df, predicted_df, on='dtime', how='outer')
    df['r_apower'] = df['r_apower'].replace('<NULL>', np.nan).astype(np.float32).interpolate()
    df.loc[df['r_apower'] < 0, 'r_apower'] = 0
    df[POWER_FEATURE] = df['r_apower'] / (STATION_CAP_MAP.get(station_id, 1) + 1e-6)
    df = df.sort_values('dtime').set_index('dtime')
    
    for key in df.columns:
        if key in [POWER_FEATURE, 'r_apower']: continue
        try:
            df[key] = pd.to_numeric(df[key], errors='coerce').interpolate()
        except: pass
    
    for variable in ALL_FEATURES:
        if variable in df.columns: df[variable].fillna(0, inplace=True)
            
    df = df.loc[~df.index.duplicated(keep='first')]
    print(f"Data shape after cleaning: {df.shape}")

    print("\n--- Splitting data into train, valid, and test sets ---")
    train_df = df[(df.index >= '2021-01-01 00:00:00') & (df.index < '2022-01-01 00:00:00')]
    initial_test_df = df[(df.index >= '2022-01-01 00:00:00') & (df.index < '2023-01-01 00:00:00')]
    
    split_point = len(initial_test_df) // 2
    valid_df = initial_test_df.iloc[:split_point]
    test_df = initial_test_df.iloc[split_point:]
    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    
    print("\n--- Fitting scaler on training data and transforming all sets ---")
    scaler = StandardScaler()
    scaler.fit(train_df[ALL_FEATURES])
    
    train_df.loc[:, ALL_FEATURES] = scaler.transform(train_df[ALL_FEATURES])
    valid_df.loc[:, ALL_FEATURES] = scaler.transform(valid_df[ALL_FEATURES])
    test_df.loc[:, ALL_FEATURES] = scaler.transform(test_df[ALL_FEATURES])
    
    print("All datasets have been transformed.")

    print("\n--- Saving all datasets to CSV files ---")
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    save_csv(train_df, 'train', station_id, columns_to_save=ALL_FEATURES)
    save_csv(valid_df, 'valid', station_id, columns_to_save=ALL_FEATURES)
    save_csv(test_df, 'test', station_id, columns_to_save=ALL_FEATURES)

    scaler_path = os.path.join(OUTPUT_PATH, f'scaler_station_{station_id}.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"\nScaler saved to: {scaler_path}")
    
    print("\n--- Typhoon Data Preprocessing and Saving Finished ---")


if __name__ == '__main__':
    if STATION_ID not in STATION_CAP_MAP:
        print(f"Error: STATION_ID {STATION_ID} is not a valid typhoon station.")
        print(f"Please choose from: {list(STATION_CAP_MAP.keys())}")
    else:
        main(STATION_ID)