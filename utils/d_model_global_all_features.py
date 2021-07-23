from utils.b_preprocessing import preprocess
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

plt.style.use('seaborn-muted')

def model_global_all_features():
    """
    Multiple linear regression with all numeric variables from the cleaned and preprocessed dataset.
    :return: the score (R² score, or accuracy) of the model and a plot
    """
    ds = preprocess()
    ds2 = ds.select_dtypes(exclude=['object']) # ds2 serves as baseline for features, first exclude categorical columns
    del ds2['price'] # the target
    del ds2['terrace_area'] # too many NaN and use of new column 'median_terrace_area' instead
    del ds2['garden_area'] # same
    del ds2['facade_count'] # same

    X = ds2
    y = ds['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1,1)
        X_test = X_test.reshape(-1,1)

    reg1 = LinearRegression()
    reg1.fit(X_train, y_train)
    reg1.score(X_train, y_train)
    y_predict = reg1.predict(X_test)
    print("Score of model on test dataset:")
    print(round(reg1.score(X_test, y_test), 2))

    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter('€{x:1.1f}M')
    plt.scatter(X_test['room_number'] - 0.1, y_test*1e-6,  color='red', label='observed')
    plt.scatter(X_test['room_number'] + 0.1, y_predict*1e-6, color='blue', linewidth=3, label='predicted')
    plt.legend()
    plt.gca().update(dict(title='Multiple linear regression global model', xlabel='X test - #rooms', ylabel='Price in M€'))
    plt.show()

    return ('Multiple Linear Regression, R² score: {:.2f}'.format(reg1.score(X_test, y_test)))
