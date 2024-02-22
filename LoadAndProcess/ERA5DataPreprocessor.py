import glob
import os
from pathlib import Path
from feature_engine.creation import CyclicalFeatures
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr
from shapely import wkt
from zope.interface import implementer

from LoadAndProcess.IDataPreprocessor import IDataPreprocessor


@implementer(IDataPreprocessor)
class ERA5DataPreprocessor:
    def process(self, is_hourly_data: bool) -> [pd.DataFrame, pd.DataFrame]:

        project_path = os.path.join(os.getcwd(), 'data/ERA5')

        # Import NetCDF ERA5 T2T 2015-2019 (5 years) 9*9 KM
        df13_2015_2019 = xr.open_dataset(os.path.join(project_path, '2015_2019.nc'))

        # Import NetCDF ERA5 Solar Radiation data 2015-2019 (5 years) 9*9 KM
        Surface_net_solar_radiation = xr.open_dataset(
            os.path.join(project_path, 'Surface_net_solar_radiation_2002_2021.nc'))
        Surface_net_thermal_radiation = xr.open_dataset(
            os.path.join(project_path, 'Surface_net_thermal_radiation_2002_2021.nc'))
        surface_solar_radiation_downwards = xr.open_dataset(
            os.path.join(project_path, 'surface_solar_radiation_downwards_2002_2021.nc'))
        Surface_thermal_radiation_downwards = xr.open_dataset(
            os.path.join(project_path, 'Surface_thermal_radiation_downwards_2002_2021.nc'))

        # Import NetCDF ERA5 Wind data 2015-2019 (5 years) 9*9KM
        Wind_2015_2017 = xr.open_dataset(os.path.join(project_path, 'Wind_2015_2017.nc'))
        Wind_2018_2020 = xr.open_dataset(os.path.join(project_path, 'Wind_2018_2020.nc'))

        # Import NetCDF MODIS LST data
        Modis_2014 = xr.open_dataset(os.path.join(project_path, 'LST_2014_NetCDF.nc')).drop(band=['1,2,4'])
        Modis_2015 = xr.open_dataset(os.path.join(project_path, 'LST_2015_NetCDF.nc')).drop(band=['1,2,4'])
        Modis_2016 = xr.open_dataset(os.path.join(project_path, 'LST_2016_NetCDF.nc')).drop(band=['1,2,4'])
        Modis_2017 = xr.open_dataset(os.path.join(project_path, 'LST_2017_NetCDF.nc')).drop(band=['1,2,4'])
        Modis_2018 = xr.open_dataset(os.path.join(project_path, 'LST_2018_NetCDF.nc')).drop(band=['1,2,4'])
        Modis_2019 = xr.open_dataset(os.path.join(project_path, 'LST_2019_NetCDF.nc')).drop(band=['1,2,4'])
        Modis_2020 = xr.open_dataset(os.path.join(project_path, 'LST_2020_NetCDF.nc')).drop(band=['1,2,4'])

        dataframe_stations = pd.read_csv(os.path.join(project_path, 'All_Staions_Locations.csv'))

        # mport MOAG staions data

        Agri_20032008 = pd.read_csv(os.path.join(project_path, 'Agri_Stations_2003_2008.csv'), parse_dates=True,
                                    dayfirst=True, skiprows=[1, 2], low_memory=False, index_col='Date & Time')
        Agri_20092014 = pd.read_csv(os.path.join(project_path, 'Agri_Stations_2008_2014.csv'), parse_dates=True,
                                    dayfirst=True, skiprows=[1, 2], low_memory=False, index_col='Date & Time')
        Agri_20152023 = pd.read_csv(os.path.join(project_path, 'Agri_Stations_2015_2023.csv'), parse_dates=True,
                                    dayfirst=True, skiprows=[1, 2], low_memory=False, index_col='Date & Time')
        Agri_locations = pd.read_csv(os.path.join(project_path, 'agri_locations.csv'))

        Agri_locations.drop(columns=['x', 'y'], inplace=True)
        Agri_locations.rename(columns={'Longtitude': 'x', 'Latitude': 'y'}, inplace=True)

        Agri_20032008.dropna(how='all', axis=1, inplace=True)
        Agri_20092014.dropna(how='all', axis=1, inplace=True)
        Agri_20152023.dropna(how='all', axis=1, inplace=True)

        # filtering MOAG stations data based on Data counting threshold
        # Concatenation of all stations for the years 2003-2023

        lst_files = [Agri_20032008, Agri_20092014, Agri_20152023]

        lst_headers = []

        new_list = []

        full_agri_df = pd.DataFrame()

        for file in lst_files:

            file.replace(['<Samp', 'NoData', 'InVld', 'Zero', 'OffScan', 'RateOC'], -999, inplace=True)

            file = file.astype(float)

            file.replace(-999, np.nan, inplace=True)

            for header in file.columns.to_list():
                header_sliced = header.split("-")[0]
                lst_headers.append(header_sliced)

            file.columns = lst_headers

            lst_headers = []

            file.reset_index(inplace=True)

            new_list.append(file)

        full_agri_df = pd.concat(new_list[1:], copy=True, axis=0, ignore_index=True)

        # >>>>>>><<<<<<<<<
        # concat all the stations value into 1 dataframe and cleaning nan based on threshhold

        perc = 30.0
        min_count = int(((100 - perc) / 100) * full_agri_df.shape[0] + 1)
        full_agri_df.dropna(axis=1, thresh=min_count, subset=None, inplace=True)
        full_agri_df.set_index('Date & Time', inplace=True)

        # <<<<<<<<<<<>>>>>>>>
        # connecting to stations locations to the stations observations values

        df_list = []
        for header in full_agri_df.columns.to_list():
            sliced = full_agri_df.loc[:, [header]]
            sliced.reset_index(inplace=True)
            sliced.rename(columns={header: 'value'}, inplace=True)
            if header in list(Agri_locations['station_name']):
                sliced['station_id'] = Agri_locations[Agri_locations['station_name'] == header].iloc[0, 0]
                sliced['station_name'] = Agri_locations[Agri_locations['station_name'] == header].iloc[0, 1]
                sliced['longitude'] = Agri_locations[Agri_locations['station_name'] == header].iloc[0, 2]
                sliced['latitude'] = Agri_locations[Agri_locations['station_name'] == header].iloc[0, 3]
                df_list.append(sliced)
        full_df_AGRI = pd.concat(df_list, axis=0, ignore_index=True)

        # Loading the corrected matrix into geoDataFrame
        ERA5_DF_with_Geometry = pd.read_csv(os.path.join(project_path, 'Zonal_Stat_results.csv'))
        ERA5_DF_with_Geometry['geometry_x'] = ERA5_DF_with_Geometry['geometry_x'].apply(wkt.loads)
        ERA5_DF_with_Geometry['Origin_Geo_poly'] = ERA5_DF_with_Geometry['Origin_Geo_poly'].apply(wkt.loads)
        ERA5_DF_with_Geometry['Geometry_ERA5_Poly'] = ERA5_DF_with_Geometry['Geometry_ERA5_Poly'].apply(wkt.loads)
        ERA5_DF_with_Geometry['Centroid'] = ERA5_DF_with_Geometry['Centroid'].apply(wkt.loads)

        Matrix_GEO = gpd.GeoDataFrame(ERA5_DF_with_Geometry)
        Matrix_GEO.set_geometry('geometry_x', inplace=True)
        Matrix_GEO = Matrix_GEO.set_crs("EPSG:3857", inplace=True)

        staions_id_10 = dataframe_stations.iloc[:]

        # Subseting the Agri_locations after cleanning the data
        Agri_locations = Agri_locations[Agri_locations['station_id'].isin(full_df_AGRI.station_id.unique().tolist())]

        # geopandas for agri station locations

        Agri_locations.rename(columns={'station_name': 'name', 'station_id': 'stationId'}, inplace=True)

        gpd_AGRI = gpd.GeoDataFrame(
            Agri_locations, geometry=gpd.points_from_xy(Agri_locations.x, Agri_locations.y)).set_crs("EPSG:3857")

        x_agri = gpd_AGRI[['x']].to_numpy().reshape(-1, 1).squeeze(axis=1)
        y_agri = gpd_AGRI[['y']].to_numpy().reshape(-1, 1).squeeze(axis=1)
        station_id_agri = gpd_AGRI[['stationId']].to_numpy().reshape(-1, 1).squeeze(axis=1)
        station_name_agri = gpd_AGRI[['name']].to_numpy().reshape(-1, 1).squeeze(axis=1)

        # geopandas for IMS station locations

        staions_id_10.rename(columns={'location.latitude': 'y', 'location.longitude': 'x'}, inplace=True)

        gpd_IMS = gpd.GeoDataFrame(
            staions_id_10, geometry=gpd.points_from_xy(staions_id_10.x, staions_id_10.y)).set_crs("EPSG:3857")

        x_ims = gpd_IMS[['x']].to_numpy().reshape(-1, 1).squeeze(axis=1)
        y_ims = gpd_IMS[['y']].to_numpy().reshape(-1, 1).squeeze(axis=1)

        # concat agri & IMS stations locations
        Concated = pd.concat([gpd_AGRI, gpd_IMS], axis=0)

        # Concated.drop(columns=['timebase','active','regionId','TD_Monitor_ID'])
        x = Concated[['x']].to_numpy().reshape(-1, 1).squeeze(axis=1)
        y = Concated[['y']].to_numpy().reshape(-1, 1).squeeze(axis=1)
        station_id_Concated = Concated[['stationId']].to_numpy().reshape(-1, 1).squeeze(axis=1)
        station_name_Concated = Concated[['name']].to_numpy().reshape(-1, 1).squeeze(axis=1)

        # allocation the weather station station XY to variables

        target_lon = xr.DataArray(x, coords={'station_id': station_id_Concated}, dims=['station_id'])
        target_lat = xr.DataArray(y, coords={'station_id': station_id_Concated}, dims=['station_id'])

        # Concat ERA5 netcdfs files based on weather stations XY coordinates to get the relevant pixels (validation data)
        Combined_Xarray = df13_2015_2019

        # Kelvice to Celciuse
        Combined_Xarray['t2m'].values = Combined_Xarray['t2m'] - 273.15

        Combined_Xarray_ERA5 = Combined_Xarray.sel(longitude=target_lon, latitude=target_lat, method='nearest').sel(
            time=slice("2015-01-01", "2020-12-31"))

        ERA5_9km_df = Combined_Xarray_ERA5.to_dataframe()
        ERA5_9km_df.drop(columns=['skt'], inplace=True)
        ERA5_9km_df.reset_index(inplace=True)

        # filter Surface_net_solar_radiation data based on target XY coordinates
        Surface_net_solar_radiation_Data = Surface_net_solar_radiation.sel(longitude=target_lon, latitude=target_lat,
                                                                           method='nearest').sel(
            time=slice("2015-01-01", "2020-12-31"))
        Surface_net_solar_radiation_df = Surface_net_solar_radiation_Data.to_dataframe()
        Surface_net_solar_radiation_df.reset_index(inplace=True)
        Surface_net_solar_radiation_df.dropna(how='any', inplace=True)

        # filter Surface_net_thermal_radiation data based on target XY coordinates
        Surface_net_thermal_radiation_Data = Surface_net_thermal_radiation.sel(longitude=target_lon,
                                                                               latitude=target_lat,
                                                                               method='nearest').sel(
            time=slice("2015-01-01", "2020-12-31"))
        Surface_net_thermal_radiation_df = Surface_net_thermal_radiation_Data.to_dataframe()
        Surface_net_thermal_radiation_df.reset_index(inplace=True)
        Surface_net_thermal_radiation_df.dropna(how='any', inplace=True)

        # filter surface_solar_radiation_downwards data based on target XY coordinates
        surface_solar_radiation_downwards_Data = surface_solar_radiation_downwards.sel(longitude=target_lon,
                                                                                       latitude=target_lat,
                                                                                       method='nearest').sel(
            time=slice("2015-01-01", "2020-12-31"))
        surface_solar_radiation_downwards_df = surface_solar_radiation_downwards_Data.to_dataframe()
        surface_solar_radiation_downwards_df.reset_index(inplace=True)
        surface_solar_radiation_downwards_df.dropna(how='any', inplace=True)

        # filter Surface_thermal_radiation_downwards data based on target XY coordinates
        Surface_thermal_radiation_downwards_Data = Surface_thermal_radiation_downwards.sel(longitude=target_lon,
                                                                                           latitude=target_lat,
                                                                                           method='nearest').sel(
            time=slice("2015-01-01", "2020-12-31"))
        Surface_thermal_radiation_downwards_df = Surface_thermal_radiation_downwards_Data.to_dataframe()
        Surface_thermal_radiation_downwards_df.reset_index(inplace=True)
        Surface_thermal_radiation_downwards_df.dropna(how='any', inplace=True)

        # filter WIND data based on target XY coordinates

        Combined_nc_wind = xr.merge([Wind_2015_2017, Wind_2018_2020], compat="no_conflicts")
        Combined_nc_wind = Combined_nc_wind.sel(longitude=target_lon, latitude=target_lat, method='nearest').sel(
            time=slice("2015-01-01", "2020-12-31"))

        Combined_nc_wind_df = Combined_nc_wind.to_dataframe()
        Combined_nc_wind_df.reset_index(inplace=True)
        Combined_nc_wind_df.drop(columns=['ssrd'], inplace=True)
        Combined_nc_wind_df.dropna(how='any', inplace=True)

        # create 2 other features: Wind Direction & Wind speed

        Combined_nc_wind_df['u10'] = (Combined_nc_wind_df['u10'] / (1000 / 3600))
        Combined_nc_wind_df['v10'] = (Combined_nc_wind_df['v10'] / (1000 / 3600))
        Combined_nc_wind_df['Wind_Speed'] = (np.power(Combined_nc_wind_df['v10'], 2) + np.power(
            Combined_nc_wind_df['u10'], 2)) ** (1 / 2)
        Combined_nc_wind_df['Wind_Direction'] = np.arctan2(Combined_nc_wind_df['v10'], Combined_nc_wind_df['u10']) * (
                180 / np.pi)

        # Filter LST NetCDF files based on Target XY points (Automatic adding the weather stations ID) and save to parquet localy (to save time)

        if not os.path.exists(os.path.join(project_path, 'Modis_parquet_files')):
            os.makedirs(os.path.join(project_path, 'Modis_parquet_files'))

        Modis_2014 = Modis_2014.sel(x=target_lon, y=target_lat, method='nearest')
        df_Modis_2014 = Modis_2014.to_dataframe().reset_index()
        df_Modis_2014.rename(columns={'__xarray_dataarray_variable__': 'ContLST_Daily'}, inplace=True)
        df_Modis_2014 = df_Modis_2014[df_Modis_2014['band'] == 2]
        df_Modis_2014.to_parquet(os.path.join(project_path, 'Modis_parquet_files/df_Modis_2014.parquet'))

        Modis_2015 = Modis_2015.sel(x=target_lon, y=target_lat, method='nearest')
        df_Modis_2015 = Modis_2015.to_dataframe().reset_index()
        df_Modis_2015.rename(columns={'__xarray_dataarray_variable__': 'ContLST_Daily'}, inplace=True)
        df_Modis_2015 = df_Modis_2015[df_Modis_2015['band'] == 2]
        df_Modis_2015.to_parquet(os.path.join(project_path, 'Modis_parquet_files/df_Modis_2015.parquet'))

        Modis_2016 = Modis_2016.sel(x=target_lon, y=target_lat, method='nearest')
        df_Modis_2016 = Modis_2016.to_dataframe().reset_index()
        df_Modis_2016.rename(columns={'__xarray_dataarray_variable__': 'ContLST_Daily'}, inplace=True)
        df_Modis_2016 = df_Modis_2016[df_Modis_2016['band'] == 2]
        df_Modis_2016.to_parquet(os.path.join(project_path, 'Modis_parquet_files/df_Modis_2016.parquet'))

        Modis_2017 = Modis_2017.sel(x=target_lon, y=target_lat, method='nearest')
        df_Modis_2017 = Modis_2017.to_dataframe().reset_index()
        df_Modis_2017.rename(columns={'__xarray_dataarray_variable__': 'ContLST_Daily'}, inplace=True)
        df_Modis_2017 = df_Modis_2017[df_Modis_2017['band'] == 2]
        df_Modis_2017.to_parquet(os.path.join(project_path, 'Modis_parquet_files/df_Modis_2017.parquet'))

        Modis_2018 = Modis_2018.sel(x=target_lon, y=target_lat, method='nearest')
        df_Modis_2018 = Modis_2018.to_dataframe().reset_index()
        df_Modis_2018.rename(columns={'__xarray_dataarray_variable__': 'ContLST_Daily'}, inplace=True)
        df_Modis_2018 = df_Modis_2018[df_Modis_2018['band'] == 2]
        df_Modis_2018.to_parquet(os.path.join(project_path, 'Modis_parquet_files/df_Modis_2018.parquet'))

        Modis_2019 = Modis_2019.sel(x=target_lon, y=target_lat, method='nearest')
        df_Modis_2019 = Modis_2019.to_dataframe().reset_index()
        df_Modis_2019.rename(columns={'__xarray_dataarray_variable__': 'ContLST_Daily'}, inplace=True)
        df_Modis_2019 = df_Modis_2019[df_Modis_2019['band'] == 2]
        df_Modis_2019.to_parquet(os.path.join(project_path, 'Modis_parquet_files/df_Modis_2019.parquet'))

        Modis_2020 = Modis_2020.sel(x=target_lon, y=target_lat, method='nearest')
        df_Modis_2020 = Modis_2020.to_dataframe().reset_index()
        df_Modis_2020.rename(columns={'__xarray_dataarray_variable__': 'ContLST_Daily'}, inplace=True)
        df_Modis_2020 = df_Modis_2020[df_Modis_2020['band'] == 2]
        df_Modis_2020.to_parquet(os.path.join(project_path, 'Modis_parquet_files/df_Modis_2020.parquet'))

        # Read all the LST parquet files from spesifice repocitiry into DataFrame

        data_dir = Path(os.path.join(project_path, 'Modis_parquet_files'))
        full_df_lst = pd.concat(
            pd.read_parquet(parquet_file)
            for parquet_file in data_dir.glob('*.parquet')
        )

        # Convert from Kelvin to celsius
        full_df_lst['ContLST_Daily'] = (full_df_lst['ContLST_Daily'] * 0.02) - 273

        full_df_lst.drop(columns=['band'], inplace=True)
        # full_df_lst.set_index('time',inplace=True)

        # upsampling to Houtly temporal resolution

        dfs_LST = []

        for st_id in list(full_df_lst.station_id.unique()):
            # data = full_df_lst[full_df_lst['station_id'] == st_id]
            # data = full_df_lst.loc[full_df_lst['station_id'] == st_id]
            data = full_df_lst[full_df_lst['station_id'] == st_id].copy()
            data['time'] = pd.to_datetime(data['time'])
            # data.loc[:, 'time'] = pd.to_datetime(data['time'])
            data.set_index(data['time'], inplace=True)
            data.drop(columns=['time'], inplace=True, errors='ignore')
            data = data.resample('H').mean().interpolate()

            data.reset_index(inplace=True)
            dfs_LST.append(data)

        LST_DF = pd.concat(dfs_LST, ignore_index=True)

        # Import NDVI files for each station, which were extracted from Google Earth Engine code editor in davance

        path = os.path.join(project_path, 'NDVI_DATA_Interpolated')
        filenames = glob.glob(path + "/*.csv")

        dfs = []
        for filename in filenames:
            file = pd.read_csv(filename, parse_dates=True, index_col=[0])
            file = file.resample('H').ffill()
            file.reset_index(inplace=True)
            dfs.append(file)

        # Concatenate all dataframes into one consolidated DataFrame
        NDVI_df = pd.concat(dfs, ignore_index=True)

        NDVI_df.rename(columns={'index': 'datetime'}, inplace=True)
        NDVI_df['datetime'] = pd.to_datetime(NDVI_df['datetime'], format='%Y/%m/%d %H:%M:%S')
        # NDVI_df=NDVI_df.set_index(['datetime','station_ID'])

        # Reading IMS data parquet (above code block)

        Full_DF_copy = pq.read_table(os.path.join(project_path, 'all_stations_data_20152020.parquet'))
        Full_DF_copy = Full_DF_copy.to_pandas()
        Full_DF_copy.drop(columns=['id', 'name'], inplace=True)

        # Changing the AGRI dataframe construnction to fit the IMS dataframe

        AGRI_stations_Val = full_df_AGRI.copy()
        AGRI_stations_Val.drop(columns=['longitude', 'latitude', 'station_name'], inplace=True)
        AGRI_stations_Val.rename(columns={'Date & Time': 'datetime', 'station_id': 'stationId'}, inplace=True)
        AGRI_stations_Val.set_index('datetime', inplace=True)
        AGRI_stations_Val.head()

        # Conbining IMS and AGRI(MOAG) data

        Full_DF_copy = pd.concat([Full_DF_copy, AGRI_stations_Val], axis=0)
        Full_DF_copy.reset_index(inplace=True)
        Full_DF_copy = Full_DF_copy.set_index(['datetime', 'stationId'])
        Hourly_IMS_values = Full_DF_copy.groupby(['stationId', (pd.Grouper(freq='1h', level=0))]).mean().reset_index()

        # combining all Features data and weather stations values into 1 big dataset

        ERA5_9KM_IMS_labels = pd.merge(Hourly_IMS_values, ERA5_9km_df, right_on=['station_id', 'time'],
                                       left_on=['stationId', 'datetime']).merge(Combined_nc_wind_df,
                                                                                on=['station_id', 'time'])
        ERA5_9KM_IMS_labels = pd.merge(ERA5_9KM_IMS_labels, NDVI_df, left_on=['stationId', 'datetime'],
                                       right_on=['station_ID', 'datetime'])
        ERA5_9KM_IMS_labels = pd.merge(ERA5_9KM_IMS_labels, Surface_net_solar_radiation_df,
                                       left_on=['stationId', 'datetime'], right_on=['station_id', 'time'],
                                       suffixes=('_ssr_x', '_ssr_y'))
        ERA5_9KM_IMS_labels = pd.merge(ERA5_9KM_IMS_labels, Surface_net_thermal_radiation_df,
                                       left_on=['stationId', 'datetime'], right_on=['station_id', 'time'],
                                       suffixes=('_sntr_x', '_sntr_y'))
        ERA5_9KM_IMS_labels = pd.merge(ERA5_9KM_IMS_labels, surface_solar_radiation_downwards_df,
                                       left_on=['stationId', 'datetime'], right_on=['station_id', 'time'],
                                       suffixes=('_ssrd_x', '_ssrd_y'))
        ERA5_9KM_IMS_labels = pd.merge(ERA5_9KM_IMS_labels, Surface_thermal_radiation_downwards_df,
                                       left_on=['stationId', 'datetime'], right_on=['station_id', 'time'],
                                       suffixes=('_strd_x', '_strd_y'))
        ERA5_9KM_IMS_labels = pd.merge(ERA5_9KM_IMS_labels, LST_DF, left_on=['stationId', 'datetime'],
                                       right_on=['station_id', 'time'])

        # Cleaning unneccesary columns

        ERA5_9KM_IMS_labels.drop(columns=['station_id_ssr_x',
                                          'time_ssr_x', 'longitude_y', 'latitude_y', 'station_id_ssr_y', 'time_ssr_y',
                                          'longitude_sntr_x', 'latitude_sntr_x',
                                          'station_id_ssrd_x', 'time_ssrd_x', 'longitude_sntr_y',
                                          'latitude_sntr_y', 'station_id_ssrd_y', 'time_ssrd_y',
                                          'longitude_strd_x', 'latitude_strd_x', 'station_id_ssrd_x', 'time_ssrd_x',
                                          'longitude_sntr_y',
                                          'latitude_sntr_y', 'station_id_ssrd_y', 'time_ssrd_y',
                                          'longitude_strd_x', 'latitude_strd_x', 'longitude_strd_y', 'latitude_strd_y',
                                          'station_id_x', 'time_x',
                                          'time_y', 'station_id_y', 'y', 'x'
                                          ], inplace=True)

        # convert dataset into to geopandas
        ERA5_GEO = gpd.GeoDataFrame(
            ERA5_9KM_IMS_labels[['t2m', 'datetime', 'stationId', 'value', 'u10', 'v10', 'Wind_Speed', 'Wind_Direction',
                                 'ssr', 'str', 'ssrd', 'strd', 'NDVI_interpo', 'ContLST_Daily'
                                 ]],
            geometry=gpd.points_from_xy(ERA5_9KM_IMS_labels.longitude_x, ERA5_9KM_IMS_labels.latitude_x)).set_crs(
            "EPSG:3857")

        # Combining the dataset with CORRECTION MATRIX values

        Join_ERA5_Matrix = gpd.sjoin_nearest(ERA5_GEO, Matrix_GEO, how='left')

        # Cleaning unneccesary columns

        Join_ERA5_Matrix.drop(columns=['__xarray_dataarray_variable__',
                                       'index_right', 'Origin_Geo_poly', 'Centroid', 't2m_x',
                                       't2m_x', 't2m_y', 'ERA5_ind', 'Geometry_ERA5_Poly',
                                       'Fixed_Temp', 'Correction_Matrix', 'Unnamed: 0'], inplace=True)

        # Build Fixed_T new feature based on mean topographice values in 9*9 and 1*1 pixels

        Join_ERA5_Matrix[
            'Topo_Mean_Diff'] = Join_ERA5_Matrix.mean_height_Modis_Pix - Join_ERA5_Matrix.mean_height_ERA5_Pix

        Correction_Matrix_Topo = 9.8 / 1000 * (
                Join_ERA5_Matrix['mean_height_Modis_Pix'] - Join_ERA5_Matrix['mean_height_ERA5_Pix'])

        Join_ERA5_Matrix['Fixed_T'] = Join_ERA5_Matrix['t2m'] - Correction_Matrix_Topo

        _Data_ = Join_ERA5_Matrix.copy()

        # extracting doy & hod from datetime

        _Data_['doy'] = _Data_['datetime'].dt.dayofyear

        _Data_['hod'] = _Data_['datetime'].dt.hour

        # exclude unreasonable values from ALL weather stations
        _Data_ = _Data_.loc[(_Data_['value'] > -15) & (_Data_['value'] < 50)]

        # exclude unreasonable values from specifice weather stations which have unreasonable values (actual values)
        _Data_.drop(_Data_[(_Data_['stationId'] == 211) & (_Data_['value'] < -5)].index, inplace=True)
        _Data_.drop(_Data_[(_Data_['stationId'] == 232) & (_Data_['value'] < -5)].index, inplace=True)
        _Data_.drop(_Data_[(_Data_['stationId'] == 355) & (_Data_['value'] < -5)].index, inplace=True)
        _Data_.drop(_Data_[(_Data_['stationId'] == 210) & (_Data_['value'] < -5)].index, inplace=True)
        _Data_.drop(_Data_[(_Data_['stationId'] == 28) & (_Data_['value'] < -5)].index, inplace=True)
        _Data_.drop(_Data_[(_Data_['stationId'] == 29) & (_Data_['value'] < -5)].index, inplace=True)
        _Data_.drop(_Data_[(_Data_['stationId'] == 42) & (_Data_['value'] < -5)].index, inplace=True)

        _Data_.dropna(how='any', inplace=True)

        # turn datetime values into cyclical features (Sin/Cos)

        X_time_vec = _Data_[['doy', 'hod']]
        cyclical = CyclicalFeatures(variables=None, drop_original=True)
        X_time_vec = cyclical.fit_transform(X_time_vec)
        _Data_ = pd.concat([_Data_, X_time_vec], axis=1)
        _Data_ = _Data_.drop(columns=['hod', 'doy'])
        _Data_.set_index(['datetime'], inplace=True)
        _Data_.drop(columns=['geometry'],inplace=True)

        if not is_hourly_data:
            _Data_ = _Data_.groupby('stationId', group_keys=False).resample('1d').mean()
            _Data_.dropna(how='any', inplace=True)
            _Data_['stationId'] = _Data_['stationId'].astype(int)
            _Data_.drop(columns=['hod_sin', 'hod_cos'], inplace=True)

        _Data_.dropna(how='any', inplace=True)
        _Data_.rename(columns={'value': 'labels'}, inplace=True)
        return _Data_.drop(columns=['stationId']), _Data_
