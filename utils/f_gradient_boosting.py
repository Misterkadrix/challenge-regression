from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

ds = pd.read_csv('/Users/cerenmorey/Desktop/BeCode/challenge-regression/data_preprocessed.csv')
def gradientboostingregressor(ds):
    '''this function splits X and y into train and test data and uses GradientBoostingRegressor()
    from sklearn to fit a model, then it uses the predict function to predict y for
    X_test and finally it determines score.'''

    ds = ds.select_dtypes(exclude=['object'])

    #Define the X feature and y
    X = ds.drop(['price', 'garden_area', 'terrace_area', 'land_surface', 'facade_count'], axis=1)
    y = ds['price']

    #Split the dataset to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    estim = GradientBoostingRegressor(n_estimators=100, max_depth=3, min_samples_split=2, learning_rate=0.1, loss='ls')
    estim.fit(X_train, y_train)
    estim.score(X_train, y_train)
    y_pred = estim.predict(X_test)
    score=estim.score(X_test, y_test)
    return ('Gradient Boosting Regressor Score: {:.2f}'.format(estim.score(X, y) * 100))


