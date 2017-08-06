import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, fbeta_score

# import xgboost algorithm
# This was installed via anaconda
# >>>> conda install -c msarahan py-xgboost
from xgboost import XGBClassifier

data = pd.read_csv("student-data.csv")

pre_data = pd.DataFrame(index=data.index)

for col, col_data in data.iteritems():

    # If data type is non-numeric, replace all yes/no values with 1/0
    if col_data.dtype == object:
        col_data = col_data.replace(['yes', 'no'], [1, 0])

    # If data type is categorical, convert to dummy variables
    if col_data.dtype == object:
        # Example: 'school' => 'school_GP' and 'school_MS'
        col_data = pd.get_dummies(col_data, prefix=col)

    # Collect the revised columns
    pre_data = pre_data.join(col_data)

features_list = list(pre_data.columns[:-1])
target_col = pre_data.columns[-1]

X_all = pre_data[features_list]
y_all = pre_data[target_col]
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.24, random_state=10, stratify=y_all)


######## XGBOOST IMPLEMENTATION #######

# Instantiate and fit the model using training data.
clf = XGBClassifier()
clf.fit(X_train, y_train)

# Predict results using testing data.
predictions = clf.predict(X_test)

# Calculate accuracy and f_score.
accuracy = accuracy_score(y_test, predictions)
f_score = fbeta_score(y_test, predictions, 0.5)
print "testing set accuracy: ", accuracy
print "testing set f-score: ", f_score

# Even using the model without any tweaks, the result is already better than the
# one achieved with tweaked Decision Tree Classifier.
# However the model takes more time to fit the data. Even so, I would prefer
# this algorithm to Decision Tree Classifier.

