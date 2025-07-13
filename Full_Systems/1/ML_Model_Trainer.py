import pandas as pd
import numpy as np
import pickle
import sklearn.ensemble as ske
from sklearn import tree, linear_model
from sklearn.feature_selection import SelectFromModel
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('data.csv', sep='|')
X = data.drop(['Name', 'md5', 'legitimate'], axis=1).values
y = data['legitimate'].values

print('Researching important feature based on %i total features\n' % X.shape[1])

# Feature selection using Extra Trees Classifier
fsel = ske.ExtraTreesClassifier().fit(X, y)
model = SelectFromModel(fsel, prefit=True)
X_new = model.transform(X)
nb_features = X_new.shape[1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2)
features = []

print('%i features identified as important:' % nb_features)

# Sort and print important features
indices = np.argsort(fsel.feature_importances_)[::-1][:nb_features]
for f in range(nb_features):
    print("%d. feature %s (%f)" % (f + 1, data.columns[2+indices[f]], fsel.feature_importances_[indices[f]]))

for f in sorted(np.argsort(fsel.feature_importances_)[::-1][:nb_features]):
    features.append(data.columns[2+f])

# Define algorithms to compare
algorithms = {
    "DecisionTree": tree.DecisionTreeClassifier(max_depth=10),
    "RandomForest": ske.RandomForestClassifier(n_estimators=50),
    "GradientBoosting": ske.GradientBoostingClassifier(n_estimators=50),
    "AdaBoost": ske.AdaBoostClassifier(n_estimators=100),
    "GNB": GaussianNB()
}

results = {}
metrics = {}
print("\nNow testing algorithms")

# Train and evaluate each algorithm
for algo in algorithms:
    clf = algorithms[algo]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Calculate metrics with zero_division parameter
    score = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    precision = precision_score(y_test, y_pred, zero_division=1)

    print("%s : Accuracy %f %%, F1 Score %f, Recall %f, Precision %f" % (algo, score*100, f1, recall, precision))
    results[algo] = score
    metrics[algo] = {'accuracy': score, 'f1': f1, 'recall': recall, 'precision': precision}

# Determine the best performing algorithm
winner = max(results, key=results.get)
print('\nWinner algorithm is %s with a %f %% success' % (winner, results[winner]*100))

# Save the best model and feature list
print('Saving algorithm and feature list in classifier directory...')
joblib.dump(algorithms[winner], 'classifier/classifier.pkl')
with open('classifier/features.pkl', 'wb') as f:  # Open in binary mode
    f.write(pickle.dumps(features))
print('Saved')

# Calculate false positive and false negative rates
clf = algorithms[winner]
res = clf.predict(X_test)
mt = confusion_matrix(y_test, res)
print("False positive rate : %f %%" % ((mt[0][1] / float(sum(mt[0])))*100))
print('False negative rate : %f %%' % ((mt[1][0] / float(sum(mt[1]))*100)))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(mt, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Plot metrics for each algorithm
metrics_df = pd.DataFrame(metrics).T
metrics_df.plot(kind='bar', figsize=(10, 7))
plt.title('Algorithm Metrics')
plt.ylabel('Score')
plt.xlabel('Algorithm')
plt.xticks(rotation=45)
plt.show()
