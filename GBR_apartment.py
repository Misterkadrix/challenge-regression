from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

def GBR_apartment(ds):
    '''this function drops House from dataset, then creates X and y and splits them  into train and test data and use GradientBoostingRegressor()
    from sklearn to fit a model then it uses the predict function to predict y for
    X_test and finally it determines score.'''
    ds_apartment = ds[ds['type'] == 'APARTMENT']


    #exclude all categorical columns from dataset
    ds_apartment_n = ds_apartment.select_dtypes(exclude=['object'])

    X_apartment = ds_apartment_n.drop(['price', 'garden_area', 'terrace_area', 'land_surface', 'facade_count'], axis=1)
    y_apartment = ds_apartment['price']

    X_apartment_train, X_apartment_test, y_apartment_train, y_apartment_test = train_test_split(X_apartment,
                                                                                                y_apartment,
                                                                                                test_size=0.2,
                                                                                                random_state=42)

    estim_a = GradientBoostingRegressor(n_estimators=200, max_depth=3, min_samples_split=2, learning_rate=0.1,
                                        loss='ls')

    estim_a.fit(X_apartment_train, y_apartment_train)


    y_pred_ap = estim_a.predict(X_apartment_test)
    estim_a_score=estim_a.score(X_apartment_test, y_apartment_test)


    return y_pred_ap  estim_a_score



