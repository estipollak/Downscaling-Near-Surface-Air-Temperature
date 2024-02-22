import os
from feature_engine.creation import CyclicalFeatures
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from shapely import wkt
from zope.interface import implementer

from LoadAndProcess.IDataPreprocessor import IDataPreprocessor


@implementer(IDataPreprocessor)
class CIMP6DataPreprocessor:
    def process(self, is_hourly_data: bool) -> [pd.DataFrame, pd.DataFrame]:

        project_path = os.path.join(os.getcwd(), 'data/CIMP6')

        Historical_data = pd.read_csv(os.path.join(project_path, 'Full_historical.csv'), parse_dates=True,
                                      dayfirst=True)
        Historical_data['time'] = pd.to_datetime(Historical_data['time'])

        MATRIX = pd.read_csv(os.path.join(project_path, 'MATRIX_27KM_1KM.csv'))

        MATRIX['geometry_x'] = MATRIX['geometry_x'].apply(wkt.loads)
        MATRIX['Origin_Geo_poly'] = MATRIX['Origin_Geo_poly'].apply(wkt.loads)
        MATRIX['Geometry_ERA5_Poly'] = MATRIX['Geometry_ERA5_Poly'].apply(wkt.loads)
        MATRIX['Centroid'] = MATRIX['Centroid'].apply(wkt.loads)

        MATRIX = gpd.GeoDataFrame(MATRIX)

        MATRIX.set_geometry('Geometry_ERA5_Poly', inplace=True)
        MATRIX.set_geometry('Origin_Geo_poly', inplace=True)
        MATRIX.set_geometry('geometry_x', inplace=True)
        MATRIX.set_geometry('Centroid', inplace=True)

        MATRIX.set_crs('epsg:4326', inplace=True)

        MATRIX['Correction_Matrix_Topo'] = 9.8 / 1000 * (MATRIX['mean_height_1KM_Pix'] - MATRIX['mean_height_27KM_Pix'])

        Agri_20032008 = pd.read_csv(os.path.join(project_path, 'Agri_Stations_2003_2008.csv'),
                                    parse_dates=True, dayfirst=True, skiprows=[1, 2], low_memory=False,
                                    index_col='Date & Time')
        Agri_20092014 = pd.read_csv(os.path.join(project_path, 'Agri_Stations_2008_2014.csv'),
                                    parse_dates=True, dayfirst=True, skiprows=[1, 2], low_memory=False,
                                    index_col='Date & Time')
        Agri_20152023 = pd.read_csv(os.path.join(project_path, 'Agri_Stations_2015_2023.csv'),
                                    parse_dates=True, dayfirst=True, skiprows=[1, 2], low_memory=False,
                                    index_col='Date & Time')
        Agri_locations = pd.read_csv(os.path.join(project_path, 'agri_locations.csv'))

        Agri_locations.drop(columns=['x', 'y'], inplace=True)
        Agri_locations.rename(columns={'Longtitude': 'x', 'Latitude': 'y'}, inplace=True)

        Agri_20032008.dropna(how='all', axis=1, inplace=True)
        Agri_20092014.dropna(how='all', axis=1, inplace=True)
        Agri_20152023.dropna(how='all', axis=1, inplace=True)

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

        dataframe_stations = pd.read_csv(os.path.join(project_path, 'All_Staions_Locations.csv'))

        # lading saved IMS data between 2008-2014

        IMS_data = pq.read_table(os.path.join(project_path, 'IMS_STATIONS_DATA_20182014.parquet'))

        IMS_data = IMS_data.to_pandas()
        IMS_data.drop(columns=['id', 'name'], inplace=True)
        IMS_data.reset_index(inplace=True)

        IMS_data = pd.merge(IMS_data,
                            dataframe_stations,
                            right_on=['stationId'],
                            left_on=['stationId'],
                            how='left')
        IMS_data.drop(columns=['timebase', 'shortName', 'regionId'], inplace=True)
        IMS_data.rename(columns={'name': 'station_name', 'y': 'latitude', 'x': 'longitude'}, inplace=True)

        IMS_data.rename(
            columns={'stationId': 'station_id'}, inplace=True)

        # resample IMS data to daily mean

        IMS_data.set_index(['datetime'], inplace=True)
        IMS_data_resampled = IMS_data.groupby('station_id').resample('1d').mean(numeric_only=True)
        IMS_data_resampled.drop(columns=['station_id'], inplace=True)
        IMS_data_resampled.reset_index(inplace=True)

        # resample MOAG(AGRI) data to daily

        full_df_AGRI.rename(columns={'Date & Time': 'datetime'}, inplace=True)
        new_cols = ['station_id', "datetime", "value", "longitude", "latitude"]
        full_df_AGRI = full_df_AGRI.reindex(columns=new_cols)
        full_df_AGRI['datetime'] = pd.to_datetime(full_df_AGRI['datetime'])

        full_df_AGRI.set_index(['datetime'], inplace=True)
        full_df_AGRI_resampled = full_df_AGRI.groupby('station_id').resample('1d').mean()
        full_df_AGRI_resampled.drop(columns=['station_id'], inplace=True)
        full_df_AGRI_resampled.reset_index(inplace=True)

        # merge IMS & MOAG
        Merged = pd.concat([IMS_data_resampled, full_df_AGRI_resampled], axis=0)
        Merged.dropna(how='all', inplace=True)

        # Merge weather stations with CMIP6 historical data

        CMIP6_Obser = pd.merge(Historical_data, Merged, right_on=['station_id', 'datetime'],
                               left_on=['stationId', 'time'])

        CMIP6_Obser['tasmax'] = CMIP6_Obser['tasmax'] - 273.15
        CMIP6_Obser['tas'] = CMIP6_Obser['tas'] - 273.15
        CMIP6_Obser['tasmin'] = CMIP6_Obser['tasmin'] - 273.15

        CMIP6_Obser.drop(columns=['Unnamed: 0'], inplace=True)

        # convert to Geopandas
        CMIP6_Obser_gpd = gpd.GeoDataFrame(
            CMIP6_Obser, geometry=gpd.points_from_xy(CMIP6_Obser.longitude, CMIP6_Obser.latitude)).set_crs("EPSG:4326")

        # define geometry
        MATRIX.geometry = MATRIX['geometry_x']

        # Spatial join with correction matrix data
        Merged_full = gpd.sjoin(CMIP6_Obser_gpd, MATRIX.set_crs("EPSG:4326"), how='left', predicate='within')
        Merged_full.dropna(how='any', inplace=True)

        # Create fixed_t feature
        Correction_Matrix_Topo = 9.8 / 1000 * (Merged_full['mean_height_1KM_Pix'] - Merged_full['mean_height_27KM_Pix'])

        Merged_full['Fixed_T'] = Merged_full['tas'] - Correction_Matrix_Topo

        # Cleaning unreasonable data (caused by gouge errors)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>..

        Merged_full = Merged_full[(Merged_full['value'] < 50) & (Merged_full['value'] > -10)]

        # >>>>>>>>>>>>>>>>>>>>>>>>>>

        _DATA_ = Merged_full.drop(columns=['Origin_Geo_poly', 'VALUE_x', 'Geometry_ERA5_Poly',
                                           'center_y', 'CMIP6_POL_ind', 'name', 'geometry', 'Unnamed: 0',
                                           '__xarray_dataarray_variable__', 'index_right'])

        # creating DOY variable/feature
        _DATA_ = _DATA_.loc[(_DATA_['value'] < 50) & (_DATA_['value'] > -20)]
        _DATA_['DOY'] = _DATA_['datetime'].dt.dayofyear
        X_time_vec = _DATA_[['DOY']]
        cyclical = CyclicalFeatures(variables=None, drop_original=True)
        X_time_vec = cyclical.fit_transform(X_time_vec)
        _DATA_ = pd.concat([_DATA_, X_time_vec], axis=1)

        # cleaning data from specific stations
        _DATA_.drop(_DATA_[(_DATA_['stationId'] == 211) & (_DATA_['value'] < -5)].index, inplace=True)
        _DATA_.drop(_DATA_[(_DATA_['stationId'] == 232) & (_DATA_['value'] < -5)].index, inplace=True)
        _DATA_.drop(_DATA_[(_DATA_['stationId'] == 355) & (_DATA_['value'] < -5)].index, inplace=True)
        _DATA_.drop(_DATA_[(_DATA_['stationId'] == 210) & (_DATA_['value'] < -5)].index, inplace=True)
        _DATA_.drop(_DATA_[(_DATA_['stationId'] == 28) & (_DATA_['value'] < -5)].index, inplace=True)
        _DATA_.drop(_DATA_[(_DATA_['stationId'] == 29) & (_DATA_['value'] < -5)].index, inplace=True)
        _DATA_.drop(_DATA_[(_DATA_['stationId'] == 42) & (_DATA_['value'] < -5)].index, inplace=True)
        _DATA_ = _DATA_.loc[~_DATA_.stationId.isin([35, 381, 380])]


        _DATA_ = _DATA_[['hurs', 'huss', 'rlds', 'rsds', 'sfcWind', 'tas', 'tasmax', 'tasmin',
                    'mean_height_27KM_Pix', 'mean_height_1KM_Pix', 'Fixed_T', 'DOY', 'DOY_sin', 'DOY_cos','value','stationId']]

        _DATA_.rename(columns={'value': 'labels'}, inplace=True)

        return _DATA_.drop(columns=['stationId']), _DATA_
