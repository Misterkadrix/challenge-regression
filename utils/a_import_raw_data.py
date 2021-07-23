import pandas as pd


# read data file houses_1.csv created in June 2021 by Solan Delvenne
# (see https://github.com/delvsola/challenge-data-analysis for the initial csv file)
def raw_and_add_admin():
    """
    Import of raw data and merge with admin. infos (city, region, province)
    :return: the raw dataset - as a pandas dataframe - with the admin info
    """
    ds = pd.read_csv('data/houses_1.csv')

    postalcode = pd.read_csv('data/postalcode.csv', encoding='latin1')
    postalcode.sort_values(by=['postalCode', 'Sous_Commune'], inplace=True)
    # Keeping only first row (corresponding to the principal municipality when postalCode ends with 0)
    # for rows having same postalCode
    postalcode.drop_duplicates(['postalCode'], keep='first', inplace=True, ignore_index=True)
    # postalcode.info()

    postalcode.rename(columns={"postalCode": "location"}, inplace=True)
    ds = pd.merge(ds, postalcode, how='left', on="location")
    print(ds.info())
    return ds
