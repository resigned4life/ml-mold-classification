import os
# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import time

# Load dataset
url = "MoldDataset30868.csv"
names = ['area', 'major-axis-length', 'minor-axis-length', 'eccentricity', 'filled-area', 'extent', 'perimeter', 'equiv-diameter', 'convex-area', 'solidity', 'class']
dataset = read_csv(url, names=names)

array = dataset.values
X = array[:, 0:10]
y = array[:, 10]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

model = LinearDiscriminantAnalysis()
name = "LDA"

#Training and Model Fit
start_time = time.time()
kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training time: {elapsed_time:.2f} seconds")

# Finalize dataset
start_time = time.time()
model.fit(X, y)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Model Fit time: {elapsed_time:.2f} seconds")


# Initialize cumulative variables
overall_accuracy = []
overall_confusion_matrix = None
overall_classification_report = ""

# Iterate through all files in a directory
testing_folder = "TestingFiles"
truth_folder = "TruthFiles"
header = ['class', 'value']

for filename in os.listdir(testing_folder):
    if filename.endswith(".csv"):
        test_file = os.path.join(testing_folder, filename)
        truth_file = os.path.join(truth_folder, filename)

        print("Processing:", test_file)

        testset = read_csv(test_file, names=names)
        truthset = read_csv(truth_file, names=header)

        testarray = testset.values
        trutharray = truthset.values

        Xnew = testarray[:, 0:10]
        ytruth = trutharray[:, 0:1]
        ynew = model.predict(Xnew)

        accuracy = accuracy_score(ytruth, ynew)
        overall_accuracy.append(accuracy)

        # Update confusion matrix
        cm = confusion_matrix(ytruth, ynew)
        if overall_confusion_matrix is None:
            overall_confusion_matrix = cm
        else:
            overall_confusion_matrix += cm

        # Update classification report
        overall_classification_report += f"\n\nClassification Report for {filename}:\n{classification_report(ytruth, ynew)}"

        # print("%s Accuracy: %.2f%%" % (filename, accuracy * 100))
        # print("Confusion Matrix:")
        # print(cm)
        # print("Classification Report:")
        # print(classification_report(ytruth, ynew))
        # print("-------------------------------------------------")

# Overall results
print('\nTraining Results for %s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
print("\nOverall Results:")
print("Overall Accuracy: %.2f%%" % (sum(overall_accuracy) / len(overall_accuracy) * 100))
print("Overall Confusion Matrix:")
print(overall_confusion_matrix)
print("Overall Classification Report:")
print(overall_classification_report)
