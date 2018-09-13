import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin


class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
                              index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)


# Load the data
train = pd.read_csv('data/shrooms.csv', header=0)
test = pd.read_csv('data/shrooms_test.csv', header=0)
'''
MANUAL Entry?
test = pd.DataFrame({
    "id": ["8124"],
    "cap-shape": ["x"],
    "cap-surface": ["s"],
    "cap-color": ["n"],
    "bruises": ["f"],
    "odor": ["n"],
    "gill-attachment": ["a"],
    "gill-spacing": ["c"],
    "gill-size": ["b"],
    "gill-color": ["y"],
    "stalk-shape": ["e"],
    "stalk-root": ["?"],
    "stalk-surface-above-ring": ["s"],
    "stalk-surface-below-ring": ["s"],
    "stalk-color-above-ring": ["o"],
    "stalk-color-below-ring": ["o"],
    "veil-type": ["p"],
    "veil-color": ["o"],
    "ring-number": ["o"],
    "ring-type": ["p"],
    "spore-print-color": ["o"],
    "population": ["c"],
    "habitat": ["l"]
})
'''

# TODO: Get useless cols (like veil-type)

# Use cols to train
# TODO: reduce to avoid overfitting?
train_cols = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
              'gill-size', 'gill-color', 'stalk-shape', 'stalk-surface-above-ring',
              'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
              'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

# Convert non numeric cols (all)
label_cols = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
              'gill-size', 'gill-color', 'stalk-shape', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
              'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
              'ring-type', 'spore-print-color', 'population', 'habitat']

big_X = train[train_cols].append(test[train_cols])
big_X_imputed = DataFrameImputer().fit_transform(big_X)

le = LabelEncoder()
for feature in label_cols:
    big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])

# Prepare the inputs for the model
train_X = big_X_imputed[0:train.shape[0]].values
test_X = big_X_imputed[train.shape[0]::].values
train_y = train['class']

# Use XGBoost
# TODO: Change to sth that supports confidence intervals
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)
predictions = gbm.predict(test_X)

# Save Results to CSV
result = pd.DataFrame({'id': test['id'], 'class': predictions})
result.to_csv("data/result.csv", index=False)

# Display Results
for index, row in result.iterrows():
    outcome = 'edible' if row['class'] == 'e' else 'poisonous'
    print('ID {} is {}'.format(row['id'], outcome))
