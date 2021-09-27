import copy
from pathlib import Path

import dill
import mlflow
import numpy as np
import torch
from torchvision.utils import make_grid
from sklearn import metrics

from trainer.base_trainer import BaseTrainer
from utils import MetricTracker, inf_loop

from logger.tensorboard_related import plot_to_image, plot_confusion_matrix, plot_convolution_filters

class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        device,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        len_epoch=None,
        initial_weights_path=None
    ):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size)) # TODO: why this value?

        self.train_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_ftns], writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            "loss", *[m.__name__ for m in self.metric_ftns], writer=self.writer
        )

        if initial_weights_path is not None:
            pretrained_weights = torch.load(initial_weights_path).get('state_dict')
            self.model.load_state_dict(pretrained_weights)



    def _train_epoch(self, epoch):  # noqa:
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()  # set model in "train" mode
        self.train_metrics.reset()  # set metrics to 0

        torch.multiprocessing.set_sharing_strategy("file_system")

        for batch_idx, (data, target) in enumerate(self.data_loader):  # for each batch
            data = copy.deepcopy(data)
            target = copy.deepcopy(target)
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()  # sets the gradients of all optimized torch.Tensor` s to zero.
            output = self.model(data)  # first forward pass
            loss = self.criterion(output, target)  # loss calculus
            loss.backward()  # Computes the gradient of current tensor w.r.t. graph leaves.
            self.optimizer.step()  # calculus for the currrent step

            #self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.writer.set_step(epoch)
            self.train_metrics.update("loss", loss.item())  # update loss metric
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))
                if self.config["mlflow"]["experiment_name"]:
                    mlflow.log_metric(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), loss.item()
                    )
                )
                # self.writer.add_image(
                #     "input", make_grid(data.cpu(), nrow=8, normalize=True)
                # )

            if batch_idx == self.len_epoch:
                break

            del data
            del target

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})
            if self.config["mlflow"]["experiment_name"]:
                for k, v in val_log.items():
                    mlflow.log_metric("val_" + k, v)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                self.valid_metrics.update("loss", loss.item())
                if self.config["mlflow"]["experiment_name"]:
                    mlflow.log_metric("loss", loss.item())

                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                    if self.config["mlflow"]["experiment_name"]:
                        mlflow.log_metric(met.__name__, met(output, target))


                if np.mod(epoch, 5) == 0:
                    pred = np.argmax(output,axis=1)
                    if batch_idx==0:
                        confusion_matrix = metrics.confusion_matrix(pred, target, labels=[0,1,2,3,4,5])
                    else:
                        confusion_matrix += metrics.confusion_matrix(pred, target, labels=[0,1,2,3,4,5])


            self.writer.set_step(epoch , "valid")
            if np.mod(epoch, 5) == 0:

                # Add validation set confusion matrix
                figure = plot_confusion_matrix(confusion_matrix, class_names=[0,1,2,3,4,5])
                cm_image = plot_to_image(figure)
                self.writer.add_image("confusion_matrix", cm_image)

                # Save filters visualization in tensorboard
                for conv_layer_name in ['conv1','conv2','conv3', 'conv4']:
                    filter_image = plot_to_image(plot_convolution_filters(self.model, conv_layer_name))
                    self.writer.add_image(f"layer {conv_layer_name}", filter_image)

                # Add histogram of model parameters to the tensorboard
                for name, p in self.model.named_parameters():
                    self.writer.add_histogram(name, p, bins="auto")

                #
                # self.writer.add_image(
                #     "input", make_grid(data.cpu(), nrow=8, normalize=True)
                # )

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        torch.save(state, filename, pickle_module=dill)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path, pickle_module=dill)
            self.logger.info("Saving current best: model_best.pth ...")
            if self.config["mlflow"]["experiment_name"]:
                mlflow.log_artifacts(Path(best_path).parent)
