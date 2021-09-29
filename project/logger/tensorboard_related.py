import io
import itertools
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import torch

def plot_convolution_filters(model, conv_layer_name):
    aux = [(name, param) for (name, param) in model.named_parameters()
           if conv_layer_name in name and 'weight' in name]
    weights = aux[0][1]

    n_out_filters, n_in_filters, h, w = weights.shape
    im_filters = np.zeros(((h+1)*n_out_filters, (w+1)*n_in_filters))
    for in_idx in range(n_in_filters):
        for out_idx in range(n_out_filters):
            im_filters[out_idx*(h+1):out_idx*(h+1)+h,
                        in_idx*(w+1):in_idx*(w+1)+w] = weights[out_idx, in_idx, :, :].detach()

    im_filters = np.expand_dims(im_filters, 2)
    # im_filters = np.concatenate(
    #     [np.concatenate([model.conv_layer_name.weight[i, :, :, :].detach(), np.zeros((1, 5, 5))]) for i in range(6)], 0)

    figure = plt.figure(figsize=(8, 8))
    plt.title(conv_layer_name)
    plt.imshow(im_filters)
    plt.colorbar()
    return figure

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues,vmin=0, vmax=1)
    plt.title("Confusion matrix")
    #plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm_normalized, decimals=2)

    # Use white text if squares are dark; otherwise black.
    # threshold = cm.max() / 2.
    threshold = 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    image = torch.tensor(np.moveaxis(np.array(Image.open(io.BytesIO(buf.getvalue())))[:,:,:3], -1, 0))
    return image
