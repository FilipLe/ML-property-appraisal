# Source Code
For full source code, access [here](https://github.com/FilipLe/ML-property-appraisal/blob/main/src/Predicting%20Property%20Valuation%20using%20Machine%20Learning.ipynb).

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
### Data Visualization
#### Scatter Plot
Categories by index:
- 0 - rooms
- 2 - Bedroom
- 3 - Bathroom
- 4 - Car
- 5 - Landsize

Change ```i[0]``` to corresponding index.
```python
import matplotlib.pyplot as plt
import seaborn as sns 

y_var = 'Price'
scatter_df = df.drop(y_var, axis = 1)
i = df3.columns

# rooms in relation price
plot1 = sns.scatterplot(i[0], y_var, data = df, color = 'orange', edgecolor = 'b', s = 150)
plt.title('{} / Price'.format(i[0]), fontsize = 16)
plt.xlabel('{}'.format(i[0]), fontsize = 14)
plt.ylabel('Price', fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.show() 
```
#### Plotting Tree
```python
import matplotlib.pyplot as plt
from sklearn import tree

plt.figure(figsize=(40,10))
tree.plot_tree(dtree, feature_names = X.columns, class_names = str(Y), max_depth = 3)
```
