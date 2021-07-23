from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


def GBR_House(ds):
    '''this function drops apartments from dataset, then creates X and y and splits them  into train and test data and use GradientBoostingRegressor()
    from sklearn to fit a model then it uses the predict function to predict y for
    X_test and finally it determines score.'''

    ds_house = ds[ds['type'] == 'HOUSE']

    #exclude all categorical columns from dataset
    ds_house_n = ds_house.select_dtypes(exclude=['object'])

    X_house = ds_house_n.drop(['price', 'garden_area', 'terrace_area', 'facade_count'], axis=1)
    y_house = ds_house['price']


    X_house_train, X_house_test, y_house_train, y_house_test = train_test_split(X_house, y_house, test_size=0.2,random_state=42)

    estim_h = GradientBoostingRegressor(n_estimators=200, max_depth=3, min_samples_split=2, learning_rate=0.1,
                                        loss='ls')

    estim_h.fit(X_house_train, y_house_train)

    y_pred_h = estim_h.predict(X_house_test)
    estim_h_score=estim_h.score(X_house_test, y_house_test)


    return ('Gradient Boosting Regressor for House Price Prediction Score: {:.2f}'.format(estim_h.score(X_house, y_house) * 100))



