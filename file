import pandas as pd
data = pd.read_csv('melb_data.csv')
print(data.head())
print(data.shape)

cleandata= data.dropna()
print(cleandata.columns)
print(cleandata.shape)


y = cleandata.Price

Features = ['Rooms', 'Distance', 'Bedroom2', \
        'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', \
'Lattitude','Longtitude']
X = cleandata[Features]

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()

print(model.fit(X,y))

from sklearn.metrics import mean_absolute_error

prediction= model.predict(X)

Error = mean_absolute_error(prediction, y)
print(Error)
