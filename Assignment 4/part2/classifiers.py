from sklearn.datasets import load_iris
from sklearn import linear_model, svm, neighbors, naive_bayes, tree, ensemble, neural_network


class ClassificationModel:
    def __init__(self, trainX, trainY):
        self.trainX = trainX
        self.trainY = trainY

    def train(self):
        self.classifier.fit(self.trainX, self.trainY)

    def predict(self, testX):
        return self.classifier.predict(testX)


class LogisticRegression(ClassificationModel):
    def __init__(self, trainX, trainY):
        super(LogisticRegression).__init__(trainX, trainY)
        self.classifier = linear_model.LogisticRegression(random_state=0)


class SVM(ClassificationModel):
    def __init__(self, trainX, trainY):
        super(SVM).__init__(trainX, trainY)
        self.classifier = svm.SVC(kernel="linear", C=0.025, random_state=101)


class KNN(ClassificationModel):
    def __init__(self, trainX, trainY):
        super(KNN).__init__(trainX, trainY)
        self.classifier = neighbors.KNeighborsClassifier(n_neighbors=15)


class NaiveBayes(ClassificationModel):
    def __init__(self, trainX, trainY):
        super(NaiveBayes).__init__(trainX, trainY)
        self.classifier = naive_bayes.GaussianNB()


class DecisionTree(ClassificationModel):
    def __init__(self, trainX, trainY):
        super(DecisionTree).__init__(trainX, trainY)
        self.classifier = tree.DecisionTreeClassifier(
            max_depth=10, random_state=101, min_samples_leaf=15, max_features=None)


class RandomForest(ClassificationModel):
    def __init__(self, trainX, trainY):
        super(RandomForest).__init__(trainX, trainY)
        self.classifier = ensemble.RandomForestClassifier(
            n_estimators=7, oob_score=True, random_state=101, max_features=None, min_samples_leaf=30)


class NeuralNetwork(ClassificationModel):
    def __init__(self, trainX, trainY):
        super(NeuralNetwork).__init__(trainX, trainY)
        self.classifier = neural_network.MLPClassifier(
            solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)


# todo
class Transformer(ClassificationModel):
    def __init__(self, trainX, trainY):
        super(Transformer).__init__(trainX, trainY)
        self.classifier = neighbors.KNeighborsClassifier(n_neighbors=15)
