from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


def gradientboostingregressor(X,y):
    '''this function split X and y into train and test data and use GradientBoostingRegressor()
    from sklearn to fit a model then it use the predict function to predict y for
    X_test and finally it determines score.'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    estim = GradientBoostingRegressor(n_estimators=100, max_depth=3, min_samples_split=2, learning_rate=0.1, loss='ls')
    estim.fit(X_train, y_train)
    estim.score(X_train, y_train)
    y_pred = estim.predict(X_test)
    score=estim.score(X_test, y_test)
    return y_pred , score
