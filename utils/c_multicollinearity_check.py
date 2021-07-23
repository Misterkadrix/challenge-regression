from utils.b_preprocessing import preprocess
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd


ds = preprocess()
ds2 = ds.select_dtypes(exclude=['object'])
#ds2.select_dtypes(exclude=['object'])
del ds2['price'] # the target
del ds2['location'] # integer but in reality the zipcode
del ds2['terrace_area'] # too many NaN
del ds2['garden_area'] # too many NaN
#del ds2['land_surface'] # too many NaN
del ds2['facade_count'] # too many NaN

X = ds2
#X = ds2.drop(ds2.columns[12:33], axis = 1) #drops dummy variables for subtype
#X = ds2.drop(ds2.columns[24:34], axis = 1) #drops dummy variables for provinces
y = ds['price']

print(X.shape)
print(y.shape)
# print(X.isna().sum()) #Check the missing values
# print(ds.groupby(['type']).land_surface.value_counts())


def calc_vif(X): # taken from https://www.analyticsvidhya.com/blog/2020/03/what-is-multicollinearity/

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)


print(calc_vif(X))
