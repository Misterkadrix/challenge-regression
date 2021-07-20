
#Convert missing values with median score for terrace and garden area, and facade count
#Creating new columns for median scores (except the land_surface column)
ds['median_terrace_area'] = np.nan
ds['median_garden_area'] = np.nan
ds['median_facade'] = np.nan

#Check median of these 3 variable

print(ds['terrace_area'].median(), ds['garden_area'].median(),ds['facade_count'].median())

#TERRACE
#Creating a median_terrace_area column: if 'terrace_area' information is available, take that, else put 16.0 as a median score.

ds['median_terrace_area']=np.where(ds['terrace']==1,ds['terrace_area'],
                                   (np.where(ds['terrace']==0,16.0, ds['median_terrace_area'])))

#If there is a terrace but the area is unknown, put 16.0 as the median score
ds['median_terrace_area'] = np.where((ds['terrace']==1 & ds['terrace_area'].isnull()) , 16.0, ds['median_terrace_area'])


#GARDEN
#Fill the column with given conditions for median garden area column
ds['median_garden_area'] = np.where(ds['garden']==1,ds['garden_area'],(np.where(ds['garden']==0,200.0, ds['median_garden_area'])))

#If there is a garden but the area is unknown, put 16.0 as the median score
ds['median_garden_area'] = np.where((ds['garden']==1 & ds['garden_area'].isnull()) , 200.0, ds['median_terrace_area'])

#FACADE
# If facade count data is available, use that, else put 2 as median score
ds['median_facade'] = np.where(ds['facade_count'].notnull(), ds['facade_count'],
                              (np.where(ds['facade_count'].isnull(), 2, ds['median_facade'])))

#Training and Test Validation Set
X =

y = target

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=41, test_size=0.2)