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



from sklearn.model_selection import train_test_split


from sklearn.metrics import mean_absolute_error
train_X, val_X, train_y, val_y= train_test_split(X,y, random_state=0)


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state = 0)
    model.fit(train_X, train_y)
    prediction = model.predict(val_X)
    Error = mean_absolute_error(prediction, val_y)
    return Error

for max_leaf_nodes in [5, 50, 500, 5000] : 
    a= get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)

    print('max leaf nodes :  %d \t\t the mean absolute error: %d' %(max_leaf_nodes ,a))

