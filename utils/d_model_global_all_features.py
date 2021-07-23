from utils.b_preprocessing import preprocess
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

plt.style.use('seaborn-muted')

ds = preprocess()

ds2 = ds.select_dtypes(exclude=['object'])
#ds2.select_dtypes(exclude=['object'])
del ds2['price'] # the target
#del ds2['location'] # integer but in reality the zipcode
del ds2['terrace_area'] # too many NaN
del ds2['garden_area'] # too many NaN
#del ds2['land_surface'] # too many NaN
del ds2['facade_count'] # too many NaN


'''
del ds2['ANVERS'] # too many NaN
del ds2['BRABANT FLAMAND'] # too many NaN
del ds2['BRABANT WALLON'] # too many NaN
del ds2['BRUXELLES'] # too many NaN
del ds2['FLANDRE-OCCIDENTALE'] # too many NaN
del ds2['FLANDRE-ORIENTALE'] # too many NaN
del ds2['HAINAUT'] # too many NaN
del ds2['LIEGE'] # too many NaN
del ds2['LIMBOURG'] # too many NaN
del ds2['LUXEMBOURG'] # too many NaN
del ds2['NAMUR'] # too many NaN
'''

X = ds2
# X = ds2[['area']]
# X = ds2[['area', 'room_number', 'swimming_pool']]
# X = ds2.drop(ds2.columns[13:34], axis = 1) #drops dummy variables for subtype

y = ds['price']

print(X.shape)
print(y.shape)
# print(X.isna().sum()) #Check the missing values


#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=772, test_size=0.2)
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

'''
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter('€{x:1.1f}M')
plt.scatter(X_test['area'], y_test*1e-6,  color='red', label='observed')
plt.scatter(X_test['area'], y_predict*1e-6, color='blue', linewidth=3, label='predicted')
plt.legend()
plt.gca().update(dict(title='Multiple linear regression global model', xlabel='X test - area in m²', ylabel='Price in M€'))
plt.show()
'''