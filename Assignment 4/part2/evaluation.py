import numpy as np 
import matplotlib.pyplot as plt 
    

def print_full_report(TN, FP, FN, TP):
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    f1 = (2*TP)/(2*TP+FP+FN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    accuracy=round(accuracy,2)
    f1=round(f1,2)
    precision=round(precision,2)
    recall=round(recall,2)

    print(f"accuracy={accuracy}, f1={f1}, precision={precision}, recall={recall}")


def report():
    print_full_report(163,36,55,146)  # LR
    print_full_report(145,54,31,170)  # SVM
    print_full_report(146,53,101,100)  # KNN
    print_full_report(127,72,69,132)  # MLP
    print_full_report(169,30,68,133)  # NaiveBayes
    print_full_report(175,24,19,182)  # DecisionTree
    print_full_report(180,19,18,183)  # RandomForest
    print_full_report(182,17,20,181)  # Transformer


if __name__ == "__main__":
    X = ['LR','SVM','KNN','MLP','NB','DT','RF','Transformer']
    accuracy = [0.77,0.79,0.61,0.65,0.76,0.89,0.91,0.91]
    f1 = [0.76,0.8,0.56,0.65,0.73,0.89,0.91,0.91]
    precision = [0.8,0.76,0.65,0.65,0.82,0.88,0.91,0.91]
    recall = [0.73,0.85,0.5,0.66,0.66,0.91,0.91,0.9]
    
    X_axis = np.arange(len(X))
    width=0.2
    
    plt.bar(X_axis + width, accuracy, width, label = 'Accuracy')
    plt.bar(X_axis + width*2, f1, width, label = 'F1')
    plt.bar(X_axis + width*3, precision, width, label = 'Precision')
    plt.bar(X_axis + width*4, recall, width, label = 'Recall')
    
    plt.xticks(X_axis+width*2.5, X)
    plt.xlabel("Models")
    # plt.ylabel("Number of Students")
    plt.title("Comparison of all models")
    plt.legend()
    plt.show()
