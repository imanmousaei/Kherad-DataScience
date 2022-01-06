import pandas as pd
import os
from sklearn.model_selection import train_test_split
import classifiers


train_path = "dataset/train.csv"
mini_train_path = "dataset/mini_train.csv"
maxn = int(1e3)


def read_dataset():
    # since my ram cant load 7GB of data (!!!) we are going we read first `maxn` rows of it.
    dataset = pd.read_csv(
        train_path,
        header=0,
        nrows=maxn,
    )
    return dataset


def read_imbalanced_dataset():
    # since dataset is imbalanced and also large(so cant load in my memory),
    # we read csv line by line. And we select first `maxn` rows with label=0
    # and first `maxn` labeled 1

    # I've tried everything I can in python but it crashed. So I have to use bash. It's also so easier
    os.system(f"head {train_path} -n 1 > {mini_train_path} ")
    os.system(f"grep 0$ {train_path} -m {maxn} >> {mini_train_path} ")
    os.system(f"grep 1$ {train_path} -m {maxn} >> {mini_train_path} ")

    dataset = pd.read_csv(mini_train_path)

    return dataset


def split_dataset(dataset: pd.DataFrame):
    # drop useless columns
    # attributed_time has data leak to test set so we remove it
    dataset.drop(['click_time', 'attributed_time'],
                 axis='columns', inplace=True)

    datasetY = dataset.loc[:, 'is_attributed']
    datasetX = dataset.drop(['is_attributed'], 'columns')

    # since test set doesnt have is_attributed column(!) we split train set into train and set.
    # because we want to calculate model's accuracy.
    trainX, testX, trainY, testY = train_test_split(
        datasetX, datasetY, test_size=0.2, random_state=42)

    print(trainY)

    return trainX, testX, trainY, testY


def main():
    dataset = read_imbalanced_dataset()
    trainX, testX, trainY, testY = split_dataset(dataset)

    models = [
        classifiers.LogisticRegression(trainX, trainY),
        classifiers.SVM(trainX, trainY),
        classifiers.KNN(trainX, trainY),
        classifiers.NaiveBayes(trainX, trainY),
        classifiers.DecisionTree(trainX, trainY),
        classifiers.RandomForest(trainX, trainY),
        classifiers.NeuralNetwork(trainX, trainY),
        classifiers.Transformer(trainX, trainY),
    ]

    print("______________________________________________________________________________________________")

    for model in models:
        report = model.report(testX, testY)
        print(model.name)
        model.plot_confusion_matrix(testX, testY)
        print(report)


if __name__ == "__main__":
    main()
