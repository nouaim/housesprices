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

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y= train_test_split(X,y, random_state=0)


print(model.fit(train_X, train_y))

from sklearn.metrics import mean_absolute_error

prediction = model.predict(val_X)

Error = mean_absolute_error(prediction, val_y)
print(Error)
