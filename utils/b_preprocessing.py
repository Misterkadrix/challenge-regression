from utils.a_import_raw_data import raw_and_add_admin
import numpy as np


def preprocess():
    '''
    Basic preprocessing of the raw dataset:
    a) dropping rows without price or area
    b) dropping duplicates
    c) computing few new features
    :return:
    '''
    ds = raw_and_add_admin()
    print('Raw data has been merged with administrative info')
    print(ds.shape)

    ds.dropna(subset=['price', 'area'], inplace=True)
    print('Data after discarding row where prices or area are unavailable')
    print(ds.shape)

    # Keeping only last row for rows having same city, price, #rooms and area
    ds.drop_duplicates(['location','type','subtype','price','room_number','area'], keep='first', inplace=True, ignore_index=True)
    print('Data after dropping duplicates zip/type/subtype/price/area/bedrooms)')
    print(ds.shape)

    # price/square meter new feature
    ds['priceSqMeter'] = ds.price/ds.area
    print('Data with additional feature')
    print(ds.shape)

    # prices between 80k€ and 2M€
    ds = ds[(80000 <= ds.price) & (ds.price <= 2e6)]

    # no grouped properties
    ds = ds[~ds['subtype'].isin(['MIXED_USE_BUILDING', 'APARTMENT_BLOCK'])]

    # bedrooms <15
    ds = ds[ds.room_number < 15]

    print('Data after complete cleaning')
    print(ds.shape)

    return ds


'''
ds = preprocess()
print(ds.iloc[:, [8, 9, 26]].head(30).to_markdown())
ds.info()
'''

ds = preprocess()
#ds.info()

#Here are the numeric Type for HOUSE AND APPARTMENT We do create a new column
ds['Type_numeric'] = ds['type'].replace({'HOUSE':0,'APARTMENT':1})
#ds['SubType_House_numeric'] = ds['subtype'].replace({'HOUSE':0,'MANSION':1,'APARTMENT':1,'BUNGALOW':1,'CASTLE':1,'CHALET':1,'COUNTY_COTTAGE':1,'DUPLEX':1,'':1})

ds.to_csv('test.csv', sep=",")
