from cProfile import label
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score


def calculate_results(x, y , model):
    y_pred  = model.predict(x)
    yy_true = (y > .5).flatten()
    yy_pred = (y_pred > .5).flatten()
    
    report = classification_report(yy_true, yy_pred, output_dict = True)

    Accuracy = accuracy_score(yy_true, yy_pred)
    Precision = report['True']['precision']
    Recall = report['True']['recall']
    F1_score = report['True']['f1-score']
    Sensitivity = Recall
    Sensificity = report['False']['recall']

    AUC = roc_auc_score(y.flatten(), y_pred.flatten())
    IOU = (Precision*Recall)/(Precision+Recall - Precision*Recall)

    results = {"Accuracy": Accuracy,
                "Precision": Precision,
                "Recall": Recall,
                "F1-score": F1_score,
                "IOU": IOU}
    return results

def plot_Curve(train_1, valid_1, train_2, valid_2, labels = ['Accuracy', 'Loss']):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes = axes.flatten()

    axes[0].plot(train_1, label = 'Training')
    axes[0].plot(valid_1, label = 'Validation')
    axes[0].set_title(labels[0]+'Curve')
    axes[0].set_xlabel('epochs')
    axes[0].set_ylabel(labels[0])
    axes[0].legend()

    axes[1].plot(train_2, label = 'Training')
    axes[1].plot(valid_2, label = 'Validation')
    axes[1].set_title(labels[1]+'Curve')
    axes[1].set_xlabel('epochs')
    axes[1].set_ylabel(labels[1])
    axes[1].legend()

def pred_and_plot(x, y, model):
    y_pred = model.predict(x)
    y_pred = y_pred > .5

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes = axes.flatten()

    axes[0].imshow(x[0])
    axes[0].set_title('Orginal Image')

    axes[1].imshow(np.squeeze(y[0], -1), cmap = 'gray')
    axes[1].set_title('Actual Masked Image')

    axes[2].imshow(np.squeeze(y_pred[0], -1), cmap = 'gray')
    axes[2].set_title("Predicted Masked Image")