# ------------------------------------------------------------------------
# Helper functions for logging and visualization of experiments.
# ------------------------------------------------------------------------

from matplotlib import pyplot as plt
import numpy as np
import sklearn.metrics as metrics


def classification_scores(Y_test, Y_test_pred, no_classes):
    """
    Computes per-class accuracy, precision, recall, and F1 score for a classification task.
    
    Args:
        Y_test: numpy.ndarray
            True labels.
        Y_test_pred: numpy.ndarray
            Predicted labels.
        no_classes: int
            Number of classes.
        
    Returns:
        tuple: A tuple containing the participant ID and per-class accuracy, precision, recall, and F1 score.
    """
    
    conf_mat = metrics.confusion_matrix(Y_test, Y_test_pred, normalize='true', labels=range(no_classes))
    # DISCLAIMER: 
    # To compute per-class accuracy, use the diagonal of the confusion matrix
    # we additionally divide by the sum of the row (ground truth) to get nan if the class is not present in the dataset
    # for that ignore zero division error only for accuracy computation
    np.seterr(divide='ignore', invalid='ignore')
    accuracy = conf_mat.diagonal() / conf_mat.sum(axis=1)
    np.seterr(divide='warn', invalid='warn')
    precision = metrics.precision_score(Y_test, Y_test_pred, zero_division=0, average=None, labels=range(no_classes))
    recall = metrics.recall_score(Y_test, Y_test_pred, zero_division=0, average=None, labels=range(no_classes))
    f1 = metrics.f1_score(Y_test, Y_test_pred, zero_division=0, average=None, labels=range(no_classes))

    return (accuracy, precision, recall, f1)
    
    
def save_confusion_matrix(y_true, y_pred, classes, path, name, normalize='true'):
    """
    Save a confusion matrix to a .png-file.
    
    Args:
        y_true: numpy.ndarray
            True labels.
        y_pred: numpy.ndarray
            Predicted labels.
        classes: list
            List of class names.
        path: str
            Path to save the confusion matrix.
        name: str
            Name of the confusion matrix.
        normalize: str
            Normalization method ('true', 'pred', or None).
        neptune_run: neptune.run
            Neptune run object to log the confusion matrix.
    """
    # set fontsize to 14
    plt.rcParams.update({'font.size': 14})
    conf_mat = metrics.confusion_matrix(y_true, y_pred, normalize=normalize, labels=range(len(classes)))
    _, ax = plt.subplots(figsize=(16, 16))
    conf_disp = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=classes) 
    # plot the confusion matrix with fontsize 14
    conf_disp.plot(ax=ax, xticks_rotation='vertical', colorbar=False)
    ax.set_title('Confusion Matrix {}'.format(name), fontsize=18)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
