from utils.a_import_raw_data import raw_and_add_admin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import tabulate

def preprocess():
    """
    Basic preprocessing of the raw dataset:
    a) dropping rows without price or area
    b) dropping duplicates
    c) computing few new features
    :return:
    """
    ds = raw_and_add_admin()
    print('Raw data has been merged with administrative info')
    print(ds.shape)

    ds.dropna(subset=['price', 'area'], inplace=True)
    print('Data after discarding row where prices or area are unavailable')
    print(ds.shape)

    # Keeping only last row for rows having same city, price, #rooms and area
    ds.drop_duplicates(['location', 'type', 'subtype', 'price', 'room_number', 'area'],
                       keep='first', inplace=True, ignore_index=True)
    print('Data after dropping duplicates (zip/type/subtype/price/area/bedrooms)')
    print(ds.shape)

    """
    # price/square meter new feature
    ds['priceSqMeter'] = ds.price/ds.area
    print('Data with additional feature')
    print(ds.shape)
    """

    # prices between 80k€ and 2M€
    ds = ds[(80000 <= ds.price) & (ds.price <= 2e6)]

    # no grouped properties
    ds = ds[~ds['subtype'].isin(['MIXED_USE_BUILDING', 'APARTMENT_BLOCK'])]

    # bedrooms <15
    ds = ds[ds.room_number < 15]


    """
    # 1. Create the encoder
    encoder = OneHotEncoder(sparse=False, drop='if_binary',  handle_unknown='ignore')
    # 2. Fit the encoder to the categorical columns
    encoder.fit(ds[['type', 'subtype', 'building_condition', 'Region', 'Province']])
    ds[['type', 'subtype', 'building_condition', 'Region', 'Province']] = encoder.transform(ds[['type', 'subtype', 'building_condition', 'Region', 'Province']])
    """

    # Convert missing values with median score for terrace and garden area, and facade count
    # Creating new columns for median scores (except the land_surface column)
    ds['median_terrace_area'] = np.nan
    ds['median_garden_area'] = np.nan
    ds['median_facade'] = np.nan

    # Check median of these 3 variable

    print(ds['terrace_area'].median(), ds['garden_area'].median(), ds['facade_count'].median())

    # TERRACE
    # Creating a median_terrace_area column: if 'terrace_area' information is available, take that, else put 16.0 as a median score.

    ds['median_terrace_area'] = np.where(ds['terrace'] == 1, ds['terrace_area'],
                                         (np.where(ds['terrace'] == 0, 16.0, ds['median_terrace_area'])))

    # If there is a terrace but the area is unknown, put 16.0 as the median score
    ds['median_terrace_area'] = np.where((ds['terrace'] == 1 & ds['terrace_area'].isnull()), 16.0,
                                         ds['median_terrace_area'])

    # GARDEN
    # Fill the column with given conditions for median garden area column
    ds['median_garden_area'] = np.where(ds['garden'] == 1, ds['garden_area'],
                                        (np.where(ds['garden'] == 0, 200.0, ds['median_garden_area'])))

    # If there is a garden but the area is unknown, put 16.0 as the median score
    ds['median_garden_area'] = np.where((ds['garden'] == 1 & ds['garden_area'].isnull()), 200.0,
                                        ds['median_terrace_area'])

    # FACADE
    # If facade count data is available, use that, else put 2 as median score
    ds['median_facade'] = np.where(ds['facade_count'].notnull(), ds['facade_count'],
                                   (np.where(ds['facade_count'].isnull(), 2, ds['median_facade'])))

    print('Data after filling with medians')
    print(ds.shape)

    ds['type_num'] = pd.get_dummies(ds.type, drop_first=True)
    subtype_num = pd.get_dummies(ds['subtype'], drop_first=False)
    building_condition_num = pd.get_dummies(ds['building_condition'], drop_first=False)
    region_num = pd.get_dummies(ds['Region'], drop_first=False)
    province_num = pd.get_dummies(ds['Province'], drop_first=False)

    ds = pd.concat([ds, subtype_num, building_condition_num, region_num, province_num], axis=1)

    print('Data after complete cleaning and preprocessing')
    print(ds.shape)

    return ds


'''
ds = preprocess()
print(ds.iloc[:, [8, 9, 26]].head(30).to_markdown())
print(ds.info())
print(ds.head(30).to_markdown())
'''

ds = preprocess()
ds.to_csv('data/data_preprocessed.csv', sep=",")
