# Source Code
For full source code, access [here blah](google.com).

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

