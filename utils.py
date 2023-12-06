from torch import save
from numpy import Inf
from os.path import exists
from os import mkdir
from torchvision import transforms

class EarlyStopping:
    def __init__(self, patience=10, delta=0, model_name="base", verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.model_name = model_name.__class__.__name__
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = Inf

    def __call__(self, val_loss, model, epoch_num):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch_num)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch_num)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss, model, epoch_num):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model at epoch {epoch_num} ...')
        
        if not exists('fruitData/checkpoints/'):
            mkdir('fruitData/checkpoints/')

        # save model with name based on model name and number of checkpoints
        save(model.state_dict(), 'fruitData/checkpoints/{}_best.pt'.format(self.model_name))
        self.val_loss_min = val_loss

def transformers(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Resize images to fit Inception V3 input size
        transforms.ToTensor()
    ])
