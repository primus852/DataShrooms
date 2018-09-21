import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Classifiers
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree

from sklearn.metrics import accuracy_score
import time
import src.helper as helper

# Init LabelEncoder
le = LabelEncoder()

# Manual Dataset to check
# Result: Edible
# x	s	y	t	a	f	c	b	k	e	c	s	s	w	w	p	w	o	p	n	n	g
manual = pd.DataFrame({
    "cap-shape": ["x"],
    "cap-surface": ["s"],
    "cap-color": ["y"],
    "bruises": ["t"],
    "odor": ["a"],
    "gill-attachment": ["f"],
    "gill-spacing": ["c"],
    "gill-size": ["b"],
    "gill-color": ["k"],
    "stalk-shape": ["e"],
    "stalk-root": ["c"],
    "stalk-surface-above-ring": ["s"],
    "stalk-surface-below-ring": ["s"],
    "stalk-color-above-ring": ["w"],
    "stalk-color-below-ring": ["w"],
    "veil-type": ["p"],
    "veil-color": ["w"],
    "ring-number": ["o"],
    "ring-type": ["p"],
    "spore-print-color": ["n"],
    "population": ["n"],
    "habitat": ["g"]
})

# LabelEncode Manual Set
manual = manual.apply(lambda x: le.fit_transform(x))

data = pd.read_csv('data/shrooms_no_header.csv', header=None)  # read data
data.rename(columns={0: 'y'}, inplace=True)  # rename predict column (edible or not)

data = data.apply(lambda x: le.fit_transform(x))  # apply LE to all columns

X = data.drop('y', 1)  # X without predict column
y = data['y']  # predict column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# todo: Adjust Hyperparameter where applicable (GradientBoosting)
test_classifiers = {
    "GradientBoosting": GradientBoostingClassifier(),
    "LogisticRegression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=1000),
    "Neural Net": MLPClassifier(alpha=1),
    "Naive Bayes": GaussianNB(),
}

# Test all classifiers
results = {}
for name, clf in list(test_classifiers.items()):
    # Train the Model and time it
    start_timer = time.process_time()
    clf.fit(X_train, y_train)
    end_timer = time.process_time()

    # Predict with the Testset
    clf_pred = clf.predict(X_test)

    # Get Total Time
    clf_time = end_timer - start_timer

    # Get the Accuracy for the Test Dataset
    test_accuracy = accuracy_score(y_test, clf_pred)

    # Get the Result for the Manual Data
    single_pred = clf.predict(manual)

    # Problem: Different Results but 100% Accuracy??? Overfitting?
    if single_pred[0] == 0:
        single = 'edible'
    else:
        single = 'poisonous'

    results[name] = {
        'classifier': name,
        'time': clf_time,
        'accuracy': test_accuracy,
        'single': single
    }

# Print the Result DataFrame and Sort by ?
helper.make_table(results)

