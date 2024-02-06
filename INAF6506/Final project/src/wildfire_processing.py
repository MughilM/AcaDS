from typing_extensions import Unpack

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=['src'],
    pythonpath=True,
)

import os
import io
import sys
import datetime as dt
from itertools import product
import time
import pprint
import sqlite3
import zipfile
import requests

import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt

import rasterio as rio
from rasterio.windows import Window


def convert_latlon_to_index(lat, lon, div_per_deg, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX):
    """
    Converts a latitude and longitude to its corresponding indices depending on the
    divisions per degree, and the minimum and maximum latitude and longitude that are
    present in the data.
    :param lat: The raw latitude value
    :param lon: The raw longitude value
    :param div_per_deg: The number of divisions per degree
    :param LAT_MIN: Minimum latitude
    :param LAT_MAX: Maximum latitude
    :param LON_MIN: Minimum longitude
    :param LON_MAX: Maximum longitude
    :return:
    """
    # For latitude...
    if lat < LAT_MIN or lat >= LAT_MAX:
        raise ValueError(f"Latitude value of {lat} is out of range ({LAT_MIN}, {LAT_MAX}]!")
    lat_index = int((lat - LAT_MAX) * -div_per_deg)
    if lon < LON_MIN or lon >= LON_MAX:
        raise ValueError(f"Longitude value of {lon} is out of range ({LON_MIN}, {LON_MAX}]!")
    lon_index = int((lon - LON_MIN) * div_per_deg)
    return lat_index, lon_index


if __name__ == '__main__':
    PREFIX = os.path.join(ROOT, 'data')
    RAW_DATA_DIR = os.path.join(PREFIX, 'raw')
    PROCESSED_DATA_DIR = os.path.join(PREFIX, 'processed')
    metrics = ['tmin', 'tmax', 'tavg', 'prec',
               'srad', 'wind', 'vapr']
    resolutions = ['10m', '5m', '2.5m', '30s']
    divs_per_deg = [6, 12, 24, 120]
    land_cover_files = {
        '10m': 'nlcd2013_10min.tif',
        '5m': 'nlcd2013_5min.tif',
        '2.5m': 'nlcd2013_2pt5min.tif',
        '30s': 'nlcd2013_30sec.tif'
    }

    # Bounding box
    LAT_MIN = 24.5
    LAT_MAX = 50
    LON_MIN = -125
    LON_MAX = -66.5

    # The query that we'll run against the fire database...
    query = """
      SELECT FIRE_YEAR, DISCOVERY_DOY, STATE, LATITUDE, LONGITUDE, FIRE_SIZE_CLASS
      FROM Fires
    """

    # Create directories if not already present...
    for res in resolutions:
        os.makedirs(os.path.join(RAW_DATA_DIR, res), exist_ok=True)
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, res), exist_ok=True)

    # List of tuples corresponding to climatology indices...
    # The method returns a 2-tuple, so we simply concatenate them together.
    # The result is a 4-tuple of lat_min_index, lon_min_index, lat_max_index, and lon_max_index.
    climate_indices = [
        convert_latlon_to_index(LAT_MIN, LON_MIN, divs, -90, 90, -180, 180) +
        convert_latlon_to_index(LAT_MAX, LON_MAX, divs, -90, 90, -180, 180)
        for divs in divs_per_deg
    ]

    start = time.perf_counter()

    # Next, we will write all the climatology files by going through each resolution.
    # In the data, increasing indices correspond to DECREASING latitudes.
    # For this reason, the index corresponding to the minimum latitude will be LARGER than that of the
    # maximum latitude, so any calculations involving height or indexing will appear switched.
    for res, divs, (LAT_MIN_IDX, LON_MIN_IDX, LAT_MAX_IDX, LON_MAX_IDX) in (
            zip(resolutions, divs_per_deg, climate_indices)):
        # Read the land cover so that we get the correct mask.
        with rio.open(os.path.join(RAW_DATA_DIR, res, land_cover_files[res]), 'r') as lc:
            lc_data = lc.read(1).astype(float)
            no_data_val = lc.nodata
            lc_mask = ~lc.dataset_mask().astype(bool)
            # With land cover, 0 is also a case for NaNs.
            lc_nan_locs = (lc_data == no_data_val) | (lc_data == 0) | lc_mask
            lc_data[lc_nan_locs] = np.nan

        # Now then, loop over all the metrics, and generate a matrix of
        # (12, <RESOLUTION>). This is the array we'll save.
        for metric in metrics:
            print(f'-{res} {metric.upper()}')
            local_folder = f'wc2.1_{res}_{metric}'
            download_url = f'http://biogeo.ucdavis.edu/data/worldclim/v2.1/base/{local_folder}.zip'
            print(f'\t- Downloading {local_folder}.zip and unzipping...', end='')
            if not os.path.exists(os.path.join(RAW_DATA_DIR, local_folder)):
                # Download the zip file, save it directly as an object and extract into directory.
                r = requests.get(download_url, stream=True)
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(os.path.join(RAW_DATA_DIR, res, local_folder))
            print('Done')

            # Next, we'll create a dictionary that will hold the data for each month.
            # We will unpack this later to save in a compressed npz file.
            # For reading the months, we will need to use rasterio.
            metric_data = []
            print('\t- Reading months...', end='')
            for i in range(1, 13):
                tif_file = f'{local_folder}_{i:02}.tif'
                with rio.open(os.path.join(RAW_DATA_DIR, res, local_folder, tif_file)) as data:
                    # Windowed reading, find the range, given the latitude and longitude indices.
                    # Because latitudes decrease as index increase, flip the min and max.
                    width = LON_MAX_IDX - LON_MIN_IDX
                    height = LAT_MIN_IDX - LAT_MAX_IDX
                    # Read the data, and also save the data mask
                    # Remember to start at max lat index, not the minimum.
                    data_array = data.read(1, window=Window(col_off=LON_MIN_IDX, row_off=LAT_MAX_IDX,
                                                            width=width, height=height)).astype(float)
                    no_data_val = data.nodata
                    data_mask = ~data.dataset_mask().astype(bool)[LAT_MAX_IDX: LAT_MIN_IDX, LON_MIN_IDX: LON_MAX_IDX]
                    # Set appropriate locations in the array to NaN
                    nan_locs = (data_array == no_data_val) | data_mask
                    data_array[nan_locs] = np.nan
                    # Append to our data array...
                    metric_data.append(data_array)
            # Stack them all to create a matrix of (12, resolution)
            metric_data = np.stack(metric_data)
            # Any locations where EITHER the land cover is NaN or the metrics are NaNs
            # should be NaN in both. Use logical_or to calculate NaNs for all the months.
            complete_nan_locs = np.logical_or.reduce(np.isnan(metric_data))
            metric_lc_nan_locs = np.logical_or(complete_nan_locs, np.isnan(lc_data))
            # Set it for our metric by broadcasting the result of np.where across all months.
            metric_data[:, Unpack[np.where(metric_lc_nan_locs)]] = np.nan
            # Update NaNs in land cover
            lc_data[np.where(metric_lc_nan_locs)] = np.nan
            print('Done')

            # SAVE THE METRIC AND LAND COVER under the processed directory.
            print(f'\t- Uploading {metric}...', end='')
            np.save(os.path.join(PROCESSED_DATA_DIR, res, f'{metric}.npy'), metric_data)
            print('Done')

        end = time.perf_counter()
        print(end - start, 'seconds.')

        # Database reading for the fire data...
        db_file = os.path.join(PREFIX, 'FPA_FOD_20170508.sqlite')
        # Establish connection
        connection = sqlite3.connect(db_file)

        # Make a cursor
        cursor = connection.cursor()

        # First, initialize the fire dataset with 0s
        # Calculate the size of the fire dataset
        height = int((LAT_MAX - LAT_MIN) * divs)
        width = int((LON_MAX - LON_MIN) * divs)
        print(width, height)
        # Create a 2d array
        fires = np.zeros((12, height, width), dtype=np.float32)

        # NOW, EXECUTE THE QUERY
        n_rows = 0
        for row in cursor.execute(query):
            # Extract the information
            year, day, state, lat, lon, size_class = row
            # If the state is Alaska (AK), Hawaii (HI) or Puerto Rico (PR), then skip it
            # Also ignore fires of size class A or B, these are really small fires.
            if state in ['AK', 'HI', 'PR'] or size_class in ['A', 'B']:
                continue
            n_rows += 1
            # Create a data object for the first day of the year, and add the proper
            # number of days according to day.
            # These are 1-indexed so we add one less.
            # Then grab the month...which is also 1-indexed.
            # Leap years are already take into account with the package...
            fire_month = (dt.date(year, 1, 1) + dt.timedelta(days=day - 1)).month

            # Calculate the correct latitude and longitude,
            # and update the count of the fires at that location.
            lat_index, lon_index = convert_latlon_to_index(lat, lon, divs, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
            fires[fire_month - 1, lat_index, lon_index] += 1

        print('N rows', n_rows)
        print('fire sum', np.sum(fires))

        # Apply the same mask as the climatology and land cover.
        # NaNs have been continuously updated in the lc_data as processed the metrics,
        # so grab it from there.
        fires[:, Unpack[np.where(np.isnan(lc_data))]] = np.nan
        print('fire sum after lc nan update', np.nansum(fires))
        np.save(os.path.join(PROCESSED_DATA_DIR, res, 'fires.npy'), fires)

        connection.close()

        # Finally, save the land cover data (which might have updated NaNs)
        np.save(os.path.join(PROCESSED_DATA_DIR, res, 'lc.npy'), lc_data)

    end = time.perf_counter()
    print(end - start, 'seconds.')

