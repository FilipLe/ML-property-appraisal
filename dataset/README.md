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
