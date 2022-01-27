import pandas as pd
data = pd.read_csv('melb_data.csv')
print(data.head())
print(data.shape)






y = data.Price

Features = ['Rooms', 'Distance', 'Bedroom2', \
        'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', \
'Lattitude','Longtitude']
X = data[Features]

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


from sklearn.metrics import mean_absolute_error
train_X, val_X, train_y, val_y= train_test_split(X,y, random_state=0)




from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score

def get_score(n_estimators):
    
    my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators,
                                                              random_state=0))
                             ])

    #model.fit(train_X, train_y)
    #prediction = model.predict(val_X)

    scores = -1 * cross_val_score(my_pipeline, train_X, train_y,
                              cv=5,
                              scoring='neg_mean_absolute_error')
    print("MAE scores:\n", scores)
    
    #Error = mean_absolute_error(prediction, val_y)
    return scores
print('score with a number of estimators 30: ',get_score(30))


a= [50,100,150,200,250,300,350,400]
results = {}
for i in range(len(a)):
    results[a[i]] = get_score(a[i]) # Your code here

import matplotlib.pyplot as plt
#%matplotlib inline

plt.plot(list(results.keys()), list(results.values()))
plt.show()
