import io
import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


def plot_convolution_filters(model, conv_layer_name):
    """
    TODO docstring à remplir
    :param model:
    :param conv_layer_name:
    :return:
    """
    aux = [(name, param) for (name, param) in model.named_parameters()
           if conv_layer_name in name and 'weight' in name]
    weights = aux[0][1]

    n_out_filters, n_in_filters, h, w = weights.shape
    im_filters = np.zeros(((h + 1) * n_out_filters, (w + 1) * n_in_filters))
    for in_idx in range(n_in_filters):
        for out_idx in range(n_out_filters):
            im_filters[out_idx * (h + 1):out_idx * (h + 1) + h,
            in_idx * (w + 1):in_idx * (w + 1) + w] = weights[out_idx, in_idx, :, :].detach()

    im_filters = np.expand_dims(im_filters, 2)
    # im_filters = np.concatenate(
    #     [np.concatenate([model.conv_layer_name.weight[i, :, :, :].detach(), np.zeros((1, 5, 5))]) for i in range(6)], 0)

    figure: Figure = plt.figure(figsize=(8, 8))
    plt.title(conv_layer_name)
    plt.imshow(im_filters)
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
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm_normalized, decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
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
    image = torch.tensor(np.moveaxis(np.array(Image.open(io.BytesIO(buf.getvalue())))[:, :, :3], -1, 0))
    return image


class Embedding():
    def __init__(self, model: nn.Module, layer_name: str, layer_dim: int = 640, outdir: str = None):
        self.model = model
        self.layer_name = layer_name
        self.layer_dim = layer_dim
        self.outdir = outdir
        if not self.outdir:
            self.outdir = Path(Path(__file__).parent.parent, "data", "embeddings")

    def clean_output(self):
        out_sprite = Path(self.outdir, "sprite.jpg")
        out_features = Path(self.outdir, "feature_vecs.tsv")
        out_metadata = Path(self.outdir, "metadata.tsv")
        out_images = Path(self.outdir, "images")
        if out_sprite.is_file():
            out_sprite.unlink()
        if out_features.is_file():
            out_features.unlink()
        if out_metadata.is_file():
            out_metadata.unlink()
        if out_images.is_dir():
            for img in out_images.glob("*.jpg"):
                img.unlink()

    def create_sprite(self, indir : str = None):
        """
        create sprite from images
        :param indir:
        :return:
        """
        if not indir:
            indir = Path(self.outdir,"images")
        outfile = Path(self.outdir, "sprite.jpg")
        if outfile.is_file():
            outfile.unlink()
        images_to_sprite(indir=indir, outdir=self.outdir)
        return True

    def get_vector(self, image: torch.Tensor):
        """
        get vector from a layer given a tensor
        :param image: Tensor representing an image
        :param model: input model
        :param layer_name: layer name
        :param layer_dim: layer dimension
        :return:
        """
        my_embedding = torch.zeros(self.layer_dim)
        layer = self.model._modules.get(self.layer_name)

        def copy_data(self, data):
            my_embedding.copy_(torch.squeeze(data[0],0))

        h = layer.register_forward_pre_hook(copy_data)
        self.model(image)
        h.remove()
        return my_embedding

    def add_data(self, data: torch.Tensor, metadata: dict):
        def safeitem(value):
            try:
                return value.item()
            except Exception as e:
                return value

        # create tsv files if they don't exist
        out_features = Path(self.outdir, "feature_vecs.tsv")
        out_metadata = Path(self.outdir, "metadata.tsv")
        import csv
        metadata_list = [dict(zip(metadata, t)) for t in zip(*metadata.values())]
        if not out_metadata.is_file():
            with open(str(out_metadata), 'w') as out_file:
                tsv_writer = csv.writer(out_file, delimiter='\t')
                tsv_writer.writerow(list(metadata.keys()))

        with open(str(out_metadata), 'a') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            for metadata_item in metadata_list:
                tsv_writer.writerow(list(map(safeitem,metadata_item.values())))

        if not out_features.is_file():
            out_features.touch(exist_ok=True)
        with open(str(out_features), 'a') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            for data_line in data:
                _data = list(map(safeitem,self.get_vector(data_line.unsqueeze(0))))
                tsv_writer.writerow(_data)
        #pictures
        pictures_tensors = metadata["tensor"]
        pictures_names = metadata["name"]
        pictures_indexes = metadata["index"]


        out_images = Path(self.outdir, "images")
        if not out_images.is_dir():
            out_images.mkdir()
        from torchvision.utils import save_image
        for (tensor,name,index) in zip(pictures_tensors,pictures_names,pictures_indexes):
            try :
                save_image(tensor, Path(out_images,f"{Path(name).stem}_{index}.jpg"))
            except:
                pass

def images_to_sprite(inlist: list[str] = None, indir: str = None, outdir: str= None):
    """
    directory of images assembled in sprite
    :param inlist: image list
    :param indir: image directory
    :param outdir: output for sample
    :return:
    """
    from PIL import Image
    if inlist:
        iconMap = inlist
    else:
        iconMap = list(map(str, Path(indir).glob("**/*.jpg")))

    images = [Image.open(filename) for filename in iconMap]
    image_width, image_height = images[0].size
    master_width = (image_width * len(images))
    master_height = image_height
    master = Image.new(
        mode='RGB',
        size=(master_width, master_height),
        color=(0, 0, 0, 0))
    for count, image in enumerate(images):
        location = image_width * count
        master.paste(image, (location, 0))
    Path(outdir).mkdir(exist_ok=True)
    master.save(str(Path(outdir, 'sprite.jpg')))

def transform_data_for_embedding(metada_file,features_file,image_dir,out_image_dir,out_metadata,out_features):
    import pandas as pd
    from pathlib import Path
    import shutil

    df = pd.read_csv(metada_file, sep="\t")
    df['line'] = df.index
    metadata_df = df.drop_duplicates(subset=['name', 'index'], keep='last')
    df = pd.read_csv(features_file, sep="\t",header=None)
    df['line'] = df.index
    feature_df = pd.merge(metadata_df["line"],df,how="inner",on="line")

    metadata_df = metadata_df.drop("line",1)
    metadata_df = metadata_df.drop("tensor",1)
    feature_df = feature_df.drop("line",1)
    metadata_df.to_csv(out_metadata, sep="\t")
    feature_df.to_csv(out_features, sep="\t",header=False)
    images = []
    for name,index in metadata_df[["name","index"]].values.tolist():
        images.append(f"{image_dir}/{Path(name).stem}_{index}.jpg")

    if not 	Path(out_image_dir).is_dir():
        Path(out_image_dir).mkdir()

    for index, item in enumerate(images):
        my_index =str(index).zfill(3)
        shutil.copy(Path(item), Path(f"{out_image_dir}/im_{my_index}.jpg"))

    #commande imagemagick
    #montage ./project/data/embeddings/my_images/im_*.jpg -tile 102x53 -geometry 102x53! sprite.jpg