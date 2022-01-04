from sklearn import linear_model, svm, neighbors, naive_bayes, tree, ensemble, neural_network, metrics
import matplotlib.pyplot as plt


class ClassificationModel:
    def __init__(self, trainX, trainY):
        self.trainX = trainX
        self.trainY = trainY

    def train(self):
        self.classifier.fit(self.trainX, self.trainY)

    def predict(self, testX):
        return self.classifier.predict(testX)

    def report(self, testX, trueY):
        self.train()
        predictY = self.predict(testX)
        # metrics.precision_recall_fscore_support
        # target_names = ['class 0', 'class 1']
        return metrics.classification_report(trueY, predictY)

    def plot_confusion_matrix(self, testX, trueY):
        predictY = self.predict(testX)
        cm = metrics.confusion_matrix(
            trueY, predictY, labels=self.classifier.classes_)
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                              display_labels=self.classifier.classes_)
        disp.plot()
        plt.show()


class LogisticRegression(ClassificationModel):
    def __init__(self, trainX, trainY):
        super().__init__(trainX, trainY)
        self.name = "LogisticRegression"
        self.classifier = linear_model.LogisticRegression(random_state=0)


class SVM(ClassificationModel):
    def __init__(self, trainX, trainY):
        super().__init__(trainX, trainY)
        self.name = "SVM"
        self.classifier = svm.SVC(kernel="linear", C=0.025, random_state=101)


class KNN(ClassificationModel):
    def __init__(self, trainX, trainY):
        super().__init__(trainX, trainY)
        self.name = "KNN"
        self.classifier = neighbors.KNeighborsClassifier(n_neighbors=20)


class NaiveBayes(ClassificationModel):
    def __init__(self, trainX, trainY):
        super().__init__(trainX, trainY)
        self.name = "NaiveBayes"
        self.classifier = naive_bayes.GaussianNB()


class DecisionTree(ClassificationModel):
    def __init__(self, trainX, trainY):
        super().__init__(trainX, trainY)
        self.name = "DecisionTree"
        self.classifier = tree.DecisionTreeClassifier(
            max_depth=10, random_state=101, min_samples_leaf=15, max_features=None)


class RandomForest(ClassificationModel):
    def __init__(self, trainX, trainY):
        super().__init__(trainX, trainY)
        self.name = "RandomForest"
        self.classifier = ensemble.RandomForestClassifier(
            n_estimators=7, oob_score=True, random_state=101, max_features=None, min_samples_leaf=30)


class NeuralNetwork(ClassificationModel):
    def __init__(self, trainX, trainY):
        super().__init__(trainX, trainY)
        self.name = "NeuralNetwork"
        self.classifier = neural_network.MLPClassifier(
            solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)


# todo
class Transformer(ClassificationModel):
    def __init__(self, trainX, trainY):
        super().__init__(trainX, trainY)
        self.name = "Transformer"
        self.classifier = neighbors.KNeighborsClassifier(n_neighbors=25)
