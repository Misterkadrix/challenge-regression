#Creating a sublist from the columns to put on a heatmap and see their correlations

for_sns = ds[['price','room_number','area','kitchen_equipped','furnished','fireplace','swimming_pool','type_num',
              'median_terrace_area','median_garden_area', 'median_facade','BRUXELLES', 'VLAANDEREN', 'WALLONIE']]

#Plot the heatmap
fig,ax = plt.subplots(figsize=(20, 10))
plot=sns.heatmap(for_sns.corr(), cmap=sns.color_palette("vlag"), annot=True)
plt.show()
figure = plot.get_figure()
plt.savefig("heatmap.png", bbox_inches="tight")

#Based on the heatmap above, create the list of columns to include in the model
for_model = ds[['room_number','area','swimming_pool','type_num','median_terrace_area','median_facade','BRUXELLES','WALLONIE']]


#Create the X and y variables
X = for_model
y = ds['price']

#Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(X_train, y_train)

y_pred_mlr= multiple_linear_regression.predict(X_test)

#Actual value and the predicted value
mlr_diff = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred_mlr})
mlr_diff.head()

#Plot the mlr_diff dataset
plt.scatter(x = 'Actual value', y = 'Predicted value', data=mlr_diff)
plt.plot(y_test, y_test, color='r',)

#Model Evaluation
from sklearn import metrics

meanAbErr = metrics.mean_absolute_error(y_test, y_pred_mlr)
meanSqErr = metrics.mean_squared_error(y_test, y_pred_mlr)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred_mlr))

print('R squared: {:.2f}'.format(multiple_linear_regression.score(X,y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)
