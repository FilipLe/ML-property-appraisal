# Dataset 

## Dataset Description

‘Melbourne Housing Snapshot’ dataset that can be found on Kaggle. 
<br><br>
The dataset consists of metrics such as number of rooms, land size, type of housing, and others for each suburb in Melbourne.
<br><br>
To view full dataset, click on the CSV file attached [../melb_data.csv](https://github.com/FilipLe/ML-property-appraisal/blob/main/dataset/melb_data.csv).

## Source

```
https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot
```

## Loading Dataset
```python
import pandas as pd
df = pd.read_csv('melb_data.csv')
```

## Data Cleaning 
Independent variable - num. of rooms, landsize, car, num. of bathrooms, num. of bedrooms
<br>Only these variables will be used to model an appropriate price.
```python
#removing unneeded labeled rows
df2 = df.drop(['Suburb','Address', 'Type', 'SellerG', 'Method', 'Date', 'Postcode', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Regionname', 'Propertycount', 'Longtitude', 'Lattitude', 'Distance'], axis=1)

#removing NULL rows
df3 = df2.dropna()
```

## Data Splitting
Splitting the data into training and testing data
```python
from sklearn.model_selection import train_test_split

X = df3.loc[:, df3.columns != 'Price']
Y = df3['Price'] > 1100000
data_train, data_test, label_train, label_test = train_test_split(X, Y, test_size=0.2, random_state = 45)
```

## Data Validation 
A portion of the training data is to be cross-validated to mitigate further effects of overfitting post data splitting.
### K - Nearest Neighbors
Cross-validation score for KNN
```python
from sklearn.model_selection import cross_val_score

max(cross_val_score(result, data_train, label_train))
```

### Decision Trees
Cross-validation score for decision tree
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

dtree = DecisionTreeClassifier(criterion="entropy", random_state=0)
dtree.fit(data_train, label_train)
dtree.score(data_test, label_test)
```
