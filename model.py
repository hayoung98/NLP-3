import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB

def Split():
    df = pd.read_csv('eigenvector.csv')
    X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    return train_X, test_X, train_y, test_y

def KNN(train_X, test_X, train_y, test_y):
    # 建立 knn 模型
    knnclassifier = KNeighborsClassifier(n_neighbors=3)
    knnclassifier.fit(train_X, train_y)
    test_y_predicted = knnclassifier.predict(test_X)
    print("Report of KNN:")
    print(classification_report(test_y, test_y_predicted, target_names=['A', 'B', 'C']))

def NBC(train_X, test_X, train_y, test_y):
    # 建立 nbc 模型
    nbclassifier = BernoulliNB()
    nbclassifier.fit(train_X, train_y)
    test_y_predicted = nbclassifier.predict(test_X)
    print("Report of NBC:")
    print(classification_report(test_y, test_y_predicted, target_names=['A', 'B', 'C']))

if __name__ == '__main__':
    train_X, test_X, train_y, test_y = Split()
    KNN(train_X, test_X, train_y, test_y)
    NBC(train_X, test_X, train_y, test_y)

    # 建立 boosting 模型
    # boost = ensemble.AdaBoostClassifier(n_estimators=100)
    # boost_fit = boost.fit(train_X, train_y)
    # test_y_predicted = boost.predict(test_X)
    # accuracy = metrics.accuracy_score(test_y, test_y_predicted)
    # print(accuracy)