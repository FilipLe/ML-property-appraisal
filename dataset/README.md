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

## Data Splitting

blha blah

```
Code: ...
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
