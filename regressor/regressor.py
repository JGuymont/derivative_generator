import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

F.mse_loss


class BaseRegressor(nn.Module):

    optimizers = {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD
    }

    loss_functions = {
        "mse": nn.MSELoss
    }

    def __init__(self, loss='mse', optimizer='adam', lr=0.001, **kargs):
        super(BaseRegressor, self).__init__()
        self.loss_function = self._get_loss_function(loss)

    def fit(self, trainloader, print_every=1, writer=None):
        """
        Train the neural network
        """
        print('Starting training...')
        print('num cont. features {}'.format(self.n_continuous_features))
        print('num cont. features {}'.format(self.n_categorical_features))
        print('num embeddings {}'.format(self.n_embeddings))
        print('input_dim: {}'.format(self.input_dim))
        print('hidden dim: {}'.format(self.hidden_dim))

        start_time = time.time()

        losses = []
        for epoch in range(self.n_epochs):
            epoch_loss = []
            for x_num, x_cat, targets in trainloader:
                self.optimizer.zero_grad()
                batch_size = x_num.shape[0]
                x_num = x_num.to(self.device)
                x_cat = x_cat.to(self.device)
                targets = targets.to(self.device).view(-1, 1)
                outputs = self.forward(x_num, x_cat)
                loss = self.loss_function(targets, outputs)
                if writer:
                    writer.add_scalar('Train/Loss', loss, epoch)
                loss.backward()
                self.optimizer.step()

                epoch_loss.append(loss.data.cpu().numpy() / batch_size)

            losses.append(np.mean(epoch_loss))

            if (epoch + 1) % print_every == 0:
                epoch_time = self._get_time(start_time, time.time())
                print('epoch: {} | loss: {:.3f} | time: {}'.format(epoch + 1, losses[-1], epoch_time))

    def _get_loss_function(self, loss: str):
        """
        Get concrete `LossFunction` object for str `loss`.
        """
        try:
            loss_class = self.loss_functions[loss]
            return loss_class()
        except KeyError:
            raise ValueError("The loss {loss} is not supported.".format(loss))

    def _get_optimizer(self, optimizer, lr):
        """
        Get concrete `Optimizer` object for str `optimizer`.
        """
        try:
            optimizer_class = self.optimizers[optimizer]
            return optimizer_class(params=self.parameters(), lr=lr)
        except KeyError:
            raise ValueError("The optimizer {optimizer} is not supported.".format(optimizer))

    def _get_time(self, starting_time, current_time):
        total_time = current_time - starting_time
        minutes = round(total_time // 60)
        seconds = round(total_time % 60)
        return '{} min., {} sec.'.format(minutes, seconds)

    def save(self, model_path_dir: str):
        """
        Save model parameters under a generated name
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optim_state_dict': self.optimizer.state_dict()
        }
        model_name = self.generate_model_name() + '.pt'
        model_path = os.path.join(model_path_dir, model_name)
        torch.save(checkpoint, model_path)
        return model_path

    def load(self, model_path: str):
        """
        Restore the model parameters
        """
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim_state_dict'])