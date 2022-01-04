import pandas as pd
import random


train_path = "dataset/train.csv"
test_path = "dataset/test.csv"
maxn = int(1e1)
p = 0.001  # select 0.1% of the lines


def read_datasets():
    # since my ram cant load 7GB of data (!!!) we are going we read p% of the rows of it.
    # if random from [0,1] interval is greater than 0.01 the row will be skipped.
    train = pd.read_csv(
        train_path,
        header=0,
        chunksize=maxn,
        # skiprows=lambda i: i > 0 and random.random() > p
    )
    # train = train.sample(maxn)
    train = pd.DataFrame(train.get_chunk(maxn))

    trainY = train.loc[:, 'is_attributed']
    trainX = train.drop(['is_attributed'], 'columns')
    

    test = pd.read_csv(
        test_path,
        header=0,
        chunksize=maxn,
        # skiprows=lambda i: i > 0 and random.random() > p
    )
    # test = test.sample(maxn)
    test = pd.DataFrame(test.get_chunk(maxn))

    testY = test.loc[:, 'is_attributed']
    testX = test.drop(['is_attributed'], 'columns')

    return trainX, trainY, testX, testY

def main():
    read_datasets()


if __name__ == "__main__":
    main()
