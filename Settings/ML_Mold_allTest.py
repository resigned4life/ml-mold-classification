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
names = ['area', 'major-axis-length', 'minor-axis-length', 'eccentricity', 'filled-area','extent','perimeter','equiv-diameter','convex-area','solidity','class']
dataset = read_csv(url, names=names)


# # shape
# print(dataset.shape)

# # head
# print(dataset.head(20))

# # descriptions
# print(dataset.describe())

# # class distribution
# print(dataset.groupby('class').size())

# # box and whisker plots
# dataset.plot(kind='box', subplots=True, layout=(5,2), sharex=False, sharey=False)
# plt.show()

# # histograms
# dataset.hist()
# plt.show()

# # scatter plot matrix
# scatter_matrix(dataset)
# plt.show()

...
# Split-out validation dataset
array = dataset.values
X = array[:,0:10]
y = array[:,10]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

...
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
 start_time = time.time()
 kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
 cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
 #cv_results_mse = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='neg_mean_squared_error')
 results.append(cv_results)
 names.append(name)
 end_time = time.time()
 print('%s: %f (%f) | Time Taken: %f' % (name, cv_results.mean(), cv_results.std(), end_time-start_time))


 # Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()