import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('data/shrooms_no_header.csv', header=None) #read data
data.rename(columns={0: 'y'}, inplace = True) #rename predict column (edible or not)

le = LabelEncoder() # encoder to do label encoder

data = data.apply(lambda x: le.fit_transform(x)) #apply LE to all columns

X = data.drop('y', 1) # X without predict column
y = data['y'] #predict column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = GradientBoostingClassifier()#you can pass arguments

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test) #it is predict for objects in test

print(accuracy_score(y_test, y_pred)) #check accuracy

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

manual = manual.apply(lambda x: le.fit_transform(x))
single_pred = clf.predict(manual)

print(single_pred)