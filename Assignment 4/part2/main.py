import pandas as pd
import random
from sklearn.model_selection import train_test_split
import classifiers
from imblearn.under_sampling import RandomUnderSampler
from csv import reader


train_path = "dataset/train.csv"
test_path = "dataset/test.csv"
maxn = int(1e3)
p = 0.001  # select 0.1% of the lines


def read_dataset():
    # since my ram cant load 7GB of data (!!!) we are going we read p% of the rows of it.
    # if random from [0,1] interval is greater than 0.01 the row will be skipped.
    dataset = pd.read_csv(
        train_path,
        header=0,
        nrows=maxn,
        # skiprows=lambda i: i > 0 and random.random() > p
    )
    # dataset = dataset.sample(maxn)
    # dataset = pd.DataFrame(dataset.get_chunk(maxn))

    return dataset


def split_dataset(dataset: pd.DataFrame):
    # drop useless columns
    # attributed_time has data leak to test set so we remove it
    dataset.drop(['click_time', 'attributed_time'], axis='columns', inplace=True)

    datasetY = dataset.loc[:, 'is_attributed']
    datasetX = dataset.drop(['is_attributed'], 'columns')

    # since test set doesnt have is_attributed column(!) we split train set into train and set. 
    # because we want to calculate model's accuracy.
    trainX, testX, trainY, testY = train_test_split(datasetX, datasetY, test_size=0.2, random_state=42)    

    print(trainY)

    return trainX, testX, trainY, testY


def main():
    dataset = read_dataset()
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

    for model in models:
        report = model.report(testX, testY)
        print(model.name,report)


if __name__ == "__main__":
    main()
