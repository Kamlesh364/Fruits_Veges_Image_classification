from torch import save, manual_seed
from numpy import Inf
from os.path import exists
from os import mkdir
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from os.path import join


# Function to transform the images
def transformers(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Resize images to fit Inception V3 input size
        transforms.ToTensor()
    ])


# Function to load the dataset
def load_dataset(folder_path, transform, batch_size=64, split_ratio=0.2, shuffle=True):
    train_dataset = datasets.ImageFolder(root=join(folder_path,'Training'), transform=transform)
    test_dataset = datasets.ImageFolder(root=join(folder_path,'Test'), transform=transform)

    class_names = train_dataset.classes # classes

    manual_seed(42) # Set random seed for reproducibility

    # Split the dataset into train and validation
    train_size = int((1-split_ratio) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create train, validation and test dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_names


# Function to load the test dataset
def load_test_dataset(folder_path, batch_size=64, img_size=224):
    test_dataset = datasets.ImageFolder(root=join(folder_path,'Test'), transform=transformers(img_size))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_dataset, test_loader


# Function to create Early Stopping
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

