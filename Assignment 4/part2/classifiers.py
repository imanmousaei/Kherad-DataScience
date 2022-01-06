from sklearn import linear_model, svm, neighbors, naive_bayes, tree, ensemble, neural_network, metrics
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow import keras

import transformer


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
        self.train()
        predictY = self.predict(testX)
        labels = ["0", "1"]
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
            solver='adam', alpha=1e-7, hidden_layer_sizes=(10, 2), random_state=101, max_iter=2000)

    def report(self, testX, trueY):
        pass


class Transformer(ClassificationModel):
    def __init__(self, trainX, trainY, vocab_size=5*10**5, columns=5, embed_dim=32, num_heads=2, ff_dim=32):
        super().__init__(trainX, trainY)
        self.name = "Transformer"
        self.vocab_size = vocab_size   # max id of inputs
        self.columns = columns  # number of features
        self.embed_dim = embed_dim  # Embedding size for each token
        self.num_heads = num_heads  # Number of attention heads
        self.ff_dim = ff_dim  # Hidden layer size in feed forward network inside transformer

        model = self.compile_model()
        self.classifier = model

    def compile_model(self):
        inputs = layers.Input(shape=(self.columns,))

        embedding_layer = transformer.TokenAndPositionEmbedding(
            self.columns, self.vocab_size, self.embed_dim)
        layer = embedding_layer(inputs)

        transformer_block = transformer.TransformerBlock(
            self.embed_dim, self.num_heads, self.ff_dim)
        
        layer = transformer_block(layer)
        layer = layers.GlobalAveragePooling1D()(layer)
        layer = layers.Dropout(0.1)(layer)
        layer = layers.Dense(20, activation="relu")(layer)
        layer = layers.Dropout(0.1)(layer)
        outputs = layers.Dense(2, activation="softmax")(layer)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        return model

    def report(self, testX, trueY):
        pass

    def train(self):
        history = self.classifier.fit(
            self.trainX, self.trainY, batch_size=32, epochs=2,
        )

    def plot_confusion_matrix(self, testX, trueY):
        self.train()
        predictY = self.predict(testX)
        labels = ["0", "1"]

        cm = metrics.confusion_matrix(trueY, predictY)

        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)

        disp.plot()
        plt.show()
