import pandas as pd
import src.helper as helper
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

# Headings
headings = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
            'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
            'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type',
            'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

# Load the data
shrooms = pd.read_csv('data/shrooms_no_header.csv', names=headings, converters={"header": float})

# Replace the ? in 'stalk-root' with 0
shrooms.loc[shrooms['stalk-root'] == '?', 'stalk-root'] = np.nan
shrooms.fillna(0, inplace=True)

# Remove columns with only one unique value
for col in shrooms.columns.values:
    if len(shrooms[col].unique()) <= 1:
        print("Removing column {}, which only contains the value: {}".format(col, shrooms[col].unique()[0]))
        shrooms.drop(col, axis=1, inplace=True)

# Col to predict later
col_predict = 'class'

# Encode per OneHotEncoding
#
# Integer for correlation matrix
all_cols_vals = shrooms.columns.values
shrooms_corr = shrooms.copy(deep=True)
helper.encode(shrooms_corr, all_cols_vals)
#
# Binary
all_cols = list(shrooms.columns.values)
all_cols.remove(col_predict)
helper.encode(shrooms, [col_predict])

# Expand Shrooms DataFrame to Binary Values
helper.expand(shrooms, all_cols)

# Display the details for the DataFrame
print(shrooms.head(10))
print(shrooms.describe())

# Show the correlation between the features (not really useful)
corr_matrix = shrooms_corr.corr()
plt.figure(figsize=(20, 16))
ax = sns.heatmap(corr_matrix, vmax=1, square=True, annot=True, fmt='.2f', cmap='GnBu', cbar_kws={"shrink": .5},
                 robust=True)
plt.title('Correlation matrix between the features', fontsize=20)
plt.show()

# Show the correlation to 'class' and sort by highest
helper.correlation_to(shrooms_corr, 'class')

# Show the cumulative explained variance
# Means "how many attributes are needed for x confidence
X = shrooms_corr[all_cols].values
X_std = StandardScaler().fit_transform(X)
pca = PCA().fit(X_std)
var_ratio = pca.explained_variance_ratio_
components = pca.components_
plt.plot(np.cumsum(var_ratio))
plt.xlim(0, 23, 1)
plt.xlabel('Number of Features', fontsize=16)
plt.ylabel('Cumulative explained variance', fontsize=16)
plt.show()

index = 0
for rat in np.cumsum(var_ratio):
    print('{} attributes: {} confidence'.format(index, rat))
    index += 1

# Remove the class we want to predict
x_all = list(shrooms.columns.values)
x_all.remove(col_predict)

# Set Train/Test ratio
ratio = 0.7

# Split the DF
df_train, df_test, X_train, Y_train, X_test, Y_test = helper.split_df(shrooms, col_predict, x_all, ratio)

# Try different classifier
# TODO: Batch Use to compare
classifier = GradientBoostingClassifier(n_estimators=1000)
# classifier = LogisticRegression()
# classifier = MLPClassifier(alpha=1)
# classifier = GaussianNB()

# TODO: Optimize Hyperparamter (where applicable)

# Time the training
timer_start = time.process_time()
classifier.fit(X_train, Y_train)
timer_stop = time.process_time()
time_diff = timer_stop - timer_start

# Get the score
score_train = classifier.score(X_train, Y_train)
score_test = classifier.score(X_test, Y_test)

print('Train Score {}, Test Score {}, Time {}'.format(score_train, score_test, time_diff))

manual = pd.DataFrame({
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


# HOW TO PREDICT?
all_cols_manual = list(manual.columns.values)
helper.encode(manual, [col_predict])
helper.expand(manual, all_cols_manual)
predictions = classifier.predict(manual)

# TODO: Test a manual DataFrame

