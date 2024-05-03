import numpy as np
import torch
import torch as t
from sklearn.metrics import f1_score, multilabel_confusion_matrix, classification_report
from tqdm.autonotebook import tqdm
import torch.optim as optim

class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda
        self._early_stopping_patience = early_stopping_patience
        self.checkpoint = None
        self.scheduler = None

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self):
        '''Overwriting checkpoint whenever validation loss is improved'''
        self.checkpoint = self._model.state_dict()

    def restore_best_checkpoint(self):
        '''Restore model weights of epoch with best validation loss'''
        self._model.load_state_dict(self.checkpoint)

    def save_onnx(self, fn):
        '''Save model as .onnx-file'''
        folder_name = 'checkpoints'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        m = self._model.cpu()
        m.eval()
        x = torch.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        torch.onnx.export(m,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          fn,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=10,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                        'output': {0: 'batch_size'}})

    def train_step(self, x, y):

        # Reset gradients to zero
        self._optim.zero_grad()

        # Forward pass: compute predicted y
        y_hat = self._model(x)

        # Compute loss
        loss = self._crit(y_hat, y)

        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # Update model parameters
        self._optim.step()

        # Return loss and predicted y
        return loss, y_hat

    def val_test_step(self, x, y):

        # Forward pass: compute predicted y
        y_hat = self._model(x)

        # Compute validation loss
        loss = self._crit(y_hat, y)

        # Return validation loss and predicted y
        return loss, y_hat

    def train_epoch(self):

        # Set training mode
        self._model.train()
        total_loss = 0
        predictions = []
        targets = []

        # Iterate over batches in the training set
        for i, batch in enumerate(self._train_dl):
            imgs, labels = batch

            # Move tensors to GPU if available
            if self._cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()

            loss, preds = self.train_step(imgs, labels)
            targets.extend(labels.cpu().detach().numpy())
            predictions.extend(preds.cpu().detach().numpy())
            total_loss += loss.item()

        # Calculate average batch loss
        avg_loss = total_loss / len(self._train_dl)

        # Calculate custom metric
        metric = self.metric_calc(np.array(targets), np.array(predictions))
        print(f"\tTraining Loss: {avg_loss}, Training Accuracy: {metric}")

        return avg_loss

    def val_test(self):

        # Set evaluation mode
        self._model.eval()
        total_loss = 0
        predictions = []
        targets = []

        # Disable gradient computation
        with torch.no_grad():

            # Iterate over batches in the training set
            for i, batch in enumerate(self._val_test_dl):
                imgs, labels = batch
                if self._cuda:
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                loss, preds = self.val_test_step(imgs, labels)
                targets.extend(labels.cpu().detach().numpy())
                predictions.extend(preds.cpu().detach().numpy())
                total_loss += loss.item()

        # Calculate average batch loss
        avg_loss = total_loss / len(self._val_test_dl)

        # Calculate custom metric
        metric = self.metric_calc(np.array(targets), np.array(predictions))
        print(f"\t Validation Loss: {avg_loss}, Validation Accuracy: {metric}")

        return avg_loss

    def metric_calc(self, targets, predictions):
        # Convert probabilities to predicted labels
        predicted_labels = np.argmax(predictions, axis=1)

        # Compare predicted labels with true labels
        correct_predictions = np.sum(predicted_labels == np.argmax(targets, axis=1))

        # Calculate accuracy
        accuracy = correct_predictions / len(targets)

        return accuracy

    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        train_losses = []
        val_losses = []
        epoch = 1
        no_improvements = 0
        epoch_best = None

        # Continue training until stopping condition is met
        while True:

            # Stop by epoch number
            if epoch > epochs:
                self.restore_best_checkpoint()
                self.save_onnx('checkpoints/checkpoint_{:03d}b.onnx'.format(epoch_best))
                print(f"The lowest validation loss was achieved in epoch{epoch_best}")
                break
            print(f'Epoch: {epoch}')

            # Train the model for one epoch and record training and validation losses
            train_losses.append(float(self.train_epoch()))
            val_losses.append(float(self.val_test()))

            # Adjust learning rate using scheduler if available
            if self.scheduler:
                self.scheduler.step(val_losses[-1])

            # Save the checkpoint if validation loss is lowest
            if val_losses[-1] == min(val_losses):
                self.save_checkpoint()
                no_improvements = 0
                epoch_best = epoch
            else:
                no_improvements += 1

            # Check if early stopping condition met
            if no_improvements >= self._early_stopping_patience:
                self.restore_best_checkpoint()
                print('Early stopping.')
                self.save_onnx('checkpoints/checkpoint_{:03d}b.onnx'.format(epoch_best))
                print(f"The lowest validation loss was achieved in epoch {epoch_best}")
                break
            epoch += 1

        return train_losses, val_losses

    def lr_scheduler(self):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self._optim, patience=5, verbose=True)