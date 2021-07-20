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
#NUMERIC subtype

#Here i got a problem with HOUSE.. can't convert to 0
#NUMERIC subtype
ds['SubType_House_numeric'] = ds['subtype'].replace({'HOUSE':0,'MANSION':1,'APARTMENT':1,'BUNGALOW':1,'CASTLE':1
,'CHALET':1,'COUNTRY_COTTAGE':1,'DUPLEX':1,'EXCEPTIONAL_PROPERTY':1,'FARMHOUSE':1,'FLAT_STUDIO':1,'GROUND_FLOOR':1
,'HOUSE':1,'KOT':1,'LOFT':1,'MANOR_HOUSE':1,'MANSION':1,'OTHER_PROPERTY':1,'PENTHOUSE':1,'SERVICE_FLAT':1
,'TOWN_HOUSE':1,'TRIPLEX':1,'VILLA':1})

#Numeric ProvinceType
#Anvers
ds['Province_Anvers_numeric'] = ds['Province'].replace({'ANVERS':0,'BRABANT FLAMAND':1,'BRABANT WALLON':1,'BRUXELLES':1
,'FLANDRE-OCCIDENTALE':1,'FLANDRE-ORIENTALE':1,'HAINAUT':1,'LIEGE':1,'LIMBOURG':1,'LUXEMBOURG':1,'NAMUR':1})
#Brabant flamand
ds['Province_BrabantFlamand_numeric'] = ds['Province'].replace({'ANVERS':1,'BRABANT FLAMAND':0,'BRABANT WALLON':1,'BRUXELLES':1
,'FLANDRE-OCCIDENTALE':1,'FLANDRE-ORIENTALE':1,'HAINAUT':1,'LIEGE':1,'LIMBOURG':1,'LUXEMBOURG':1,'NAMUR':1})
#Brabant wallon
ds['Province_BrabantWallon_numeric'] = ds['Province'].replace({'ANVERS':1,'BRABANT FLAMAND':1,'BRABANT WALLON':0,'BRUXELLES':1
,'FLANDRE-OCCIDENTALE':1,'FLANDRE-ORIENTALE':1,'HAINAUT':1,'LIEGE':1,'LIMBOURG':1,'LUXEMBOURG':1,'NAMUR':1})
#Bruxelles
ds['Province_Bruxelles_numeric'] = ds['Province'].replace({'ANVERS':1,'BRABANT FLAMAND':1,'BRABANT WALLON':1,'BRUXELLES':0
,'FLANDRE-OCCIDENTALE':1,'FLANDRE-ORIENTALE':1,'HAINAUT':1,'LIEGE':1,'LIMBOURG':1,'LUXEMBOURG':1,'NAMUR':1})
#FlandreOccidentale
ds['Province_FlandreOccidentale_numeric'] = ds['Province'].replace({'ANVERS':1,'BRABANT FLAMAND':1,'BRABANT WALLON':1,'BRUXELLES':1
,'FLANDRE-OCCIDENTALE':0,'FLANDRE-ORIENTALE':1,'HAINAUT':1,'LIEGE':1,'LIMBOURG':1,'LUXEMBOURG':1,'NAMUR':1})
#Flandre Orientale
ds['Province_FlandreOrientale_numeric'] = ds['Province'].replace({'ANVERS':1,'BRABANT FLAMAND':1,'BRABANT WALLON':1,'BRUXELLES':1
,'FLANDRE-OCCIDENTALE':1,'FLANDRE-ORIENTALE':0,'HAINAUT':1,'LIEGE':1,'LIMBOURG':1,'LUXEMBOURG':1,'NAMUR':1})
#Hainaut
ds['Province_Hainaut_numeric'] = ds['Province'].replace({'ANVERS':1,'BRABANT FLAMAND':1,'BRABANT WALLON':1,'BRUXELLES':1
,'FLANDRE-OCCIDENTALE':1,'FLANDRE-ORIENTALE':1,'HAINAUT':0,'LIEGE':1,'LIMBOURG':1,'LUXEMBOURG':1,'NAMUR':1})
#Liege
ds['Province_LIEGE_numeric'] = ds['Province'].replace({'ANVERS':1,'BRABANT FLAMAND':1,'BRABANT WALLON':1,'BRUXELLES':1
,'FLANDRE-OCCIDENTALE':1,'FLANDRE-ORIENTALE':1,'HAINAUT':1,'LIEGE':0,'LIMBOURG':1,'LUXEMBOURG':1,'NAMUR':1})
#Limbourg
ds['Province_Limbourg_numeric'] = ds['Province'].replace({'ANVERS':1,'BRABANT FLAMAND':1,'BRABANT WALLON':1,'BRUXELLES':1
,'FLANDRE-OCCIDENTALE':1,'FLANDRE-ORIENTALE':1,'HAINAUT':1,'LIEGE':1,'LIMBOURG':0,'LUXEMBOURG':1,'NAMUR':1})
#Luxembourg
ds['Province_Luxembourg_numeric'] = ds['Province'].replace({'ANVERS':1,'BRABANT FLAMAND':1,'BRABANT WALLON':1,'BRUXELLES':1
,'FLANDRE-OCCIDENTALE':1,'FLANDRE-ORIENTALE':1,'HAINAUT':1,'LIEGE':1,'LIMBOURG':1,'LUXEMBOURG':0,'NAMUR':1})
#Namur
ds['Province_Namur_numeric'] = ds['Province'].replace({'ANVERS':1,'BRABANT FLAMAND':1,'BRABANT WALLON':1,'BRUXELLES':1
,'FLANDRE-OCCIDENTALE':1,'FLANDRE-ORIENTALE':1,'HAINAUT':1,'LIEGE':1,'LIMBOURG':1,'LUXEMBOURG':1,'NAMUR':0})

#Here we will add numeric Columns for the Region Columns

ds['Region_Bruxelles_numeric'] = ds['Region'].replace({'BRUXELLES':0,'VLAANDEREN':1,'WALLONIE':1})
ds['Region_Vlaanderen_numeric'] = ds['Region'].replace({'BRUXELLES':1,'VLAANDEREN':0,'WALLONIE':1})
ds['Region_Wallonie_numeric'] = ds['Region'].replace({'BRUXELLES':1,'VLAANDEREN':1,'WALLONIE':0})





ds.to_csv('test.csv', sep=",")
