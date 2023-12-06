"""
python your_script.py --dataset_path your_dataset_path --model_name your_model_name --epochs 10 --batch_size 64 --split_ratio 0.2 --shuffle True --lr 0.001 --patience 50
"""

from utils import *
from argparse import ArgumentParser
from os.path import exists
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from os.path import join


# Function to load the dataset
def load_dataset(folder_path, transform, batch_size=64, split_ratio=0.2, shuffle=True):
    train_dataset = datasets.ImageFolder(root=join(folder_path,'Training'), transform=transform)
    test_dataset = datasets.ImageFolder(root=join(folder_path,'Test'), transform=transform)

    class_names = train_dataset.classes # classes

    torch.manual_seed(42) # Set random seed for reproducibility

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


# Function to load the model
def model_selection(model_name):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model
    if model_name == 'base':
        from baselinemodel import BaselineModel
        model = BaselineModel(num_classes=len(class_names)).to(device)
        imf_size = 100
    
    elif model_name == 'alexnet':
        from torchvision.models import alexnet
        model = alexnet(weights='AlexNet_Weights.IMAGENET1K_V1')
        model.classifier[6] = nn.Linear(4096, len(class_names))
        model = model.to(device)
        img_size = 224
    
    elif model_name == 'inception':
        from torchvision.models import inception_v3
        model = inception_v3(weights='Inception_V3_Weights.IMAGENET1K_V1')
        model.fc = nn.Linear(2048, len(class_names))
        model = model.to(device)
        img_size = 299

    elif model_name == 'mobilenet':
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(1280, len(class_names))
        model = model.to(device)
        img_size = 299
    
    elif model_name == 'custom_mobilenet':
        from custommobilenet import mobilenet_v2
        model = mobilenet_v2(num_classes=len(class_names)).to(device)
        img_size = 299

    return model, img_size


# Function to evaluate the model
def evaluate_model(model, test_loader, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define the device

    # load best trained weights
    try:
        model.load_state_dict(torch.load('fruitData/checkpoints/{}_best.pt'.format(model.__class__.__name__)))
    except:
        pass

    # Evaluate the model on test set
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f'Overall Test Accuracy: {accuracy*100:.4f} %')

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    # Calculate accuracy for each class
    class_labels = np.array(all_labels)
    class_predictions = np.array(all_predictions)
    class_accuracy = {class_name: accuracy_score(class_labels[class_labels == i], 
                                    class_predictions[class_labels == i]) for i, class_name in enumerate(class_names)}
    
    # Calculate and plot misclassifications heatmap
    misclassifications = conf_matrix - np.diag(np.diag(conf_matrix))

    return accuracy, conf_matrix, class_accuracy, {"class_labels": class_labels, "class_predictions": class_predictions}, misclassifications


# Function to train the model
def train_model(dataset_path='fruitData', model_name='mobilenet', epochs=10, batch_size=64, 
                split_ratio=0.2, shuffle=True, learning_rate=0.001, patience=50, evaluate=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define the device

    # Define the model
    model, img_size = model_selection(model_name)

    # Load the dataset
    train_loader, val_loader, test_loader, class_names = load_dataset(dataset_path, transformers(img_size), batch_size, split_ratio, shuffle)
        
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping
    early_stopping = EarlyStopping(patience=patience, model_name=model.__class__.__name__, verbose=True)

    # Lists to store loss and accuracy values for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []  # New list for training accuracies
    val_accuracies = []

    # Training loop
    num_epochs = epochs

    # print logs
    print(f"Training {model.__class__.__name__} for following parameters-\n",
            f"Number of epochs: {epochs}\n",
            f"Batch size: {batch_size}\n",
            f"Learning rate: {learning_rate}\n",
            f"Patience: {patience}\n",
            f"Shuffle: {shuffle}\n",
            f"Split ratio: {split_ratio}\n",
            f"Dataset path: {dataset_path}\n",
            f"Optimizer: Adam\n",
            f"Loss function: Cross Entropy Loss\n",)
    
    print("Training Started...\n","...might take a few minutes to train one epoch depending on your hardware.\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted_train = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        # Calculate average training loss and accuracy for the epoch
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Validation loop
        model.eval()
        with torch.no_grad():
            correct_val = 0
            total_val = 0
            val_loss = 0.0
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted_val = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted_val == labels).sum().item()

            # Calculate average validation loss and accuracy for the epoch
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            val_accuracy = correct_val / total_val
            val_accuracies.append(val_accuracy)

            # Early stopping check
            if early_stopping(val_loss, model, epoch_num=epoch + 1):
                print("Early stopping!")
                break

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    print('Finished Training')

    # Evaluate the model on test set
    if evaluate:
        test_accuracy, conf_matrix, class_accuracy, class_info, misclassifications = evaluate_model(model, test_loader, class_names)

    losses = {"train_losses": train_losses, "val_losses": val_losses}
    accuracies = {"train_accuracies": train_accuracies, "val_accuracies": val_accuracies}
    evaluation_results = {"test_accuracy": test_accuracy, "conf_matrix": conf_matrix, 
                          "class_accuracy": class_accuracy, "class_labels": class_info["class_labels"],
                          "class_predictions": class_info["class_predictions"], "misclassifications": misclassifications}

    return losses, accuracies, model, evaluation_results, class_names

if __name__ == "__main__":
    # Parse command line arguments
    parser = ArgumentParser(description='Train a deep learning model.')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--dataset_path', type=str, default='fruitData', help='/path/to/parent/folder/of/train/test/datasets')
    parser.add_argument('--model_name', type=str, default='mobilenet', help='Name of the model--(base, alexnet, inception, mobilenet, custom_mobilenet)')
    parser.add_argument('--checkpoints', type=str, default='fruitData/checkpoints', help='path to saved checkpoints')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--split_ratio', type=float, default=0.2, help='Ratio for train-validation split')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle the dataset before splitting')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
    parser.add_argument('--evaluate', type=bool, default=True, help='Evaluate the model on test set')
    parser.add_argument('--show_plots', type=bool, default=True, help='Show plots')

    args = parser.parse_args()

    # Check if the provided dataset path exists
    if not exists(args.dataset_path):
        print(f"Error: Dataset path '{args.dataset_path}' not found.")
        exit(1)

    else:# Train the model
        if args.mode == 'train':
            losses, accuracies, model, evaluation_results, class_names = train_model(dataset_path=args.dataset_path, 
                        model_name=args.model_name, epochs=args.epochs, batch_size=args.batch_size, 
                        split_ratio=args.split_ratio, shuffle=args.shuffle, learning_rate=args.lr, 
                        patience=args.patience, evaluate=args.evaluate)
            
            # plot the training and validation losses and accuracies
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 2, 1)
            plt.plot(losses['train_losses'], label='Training Loss', alpha=0.7)
            plt.plot(losses["val_losses"], label='Validation Loss', alpha=0.7)
            plt.title('Losses')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(accuracies["train_accuracies"], label='Training Accuracy', color='green', alpha=0.7)
            plt.plot(accuracies["val_accuracies"], label='Validation Accuracy', color='orange', alpha=0.7)
            plt.title('Accuracies')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()

            # Save the plot as a PNG with a transparent background
            plt.savefig("{}.png".format(model.__class__.__name__), format="png")

            # Show the plot (optional)
            if args.show_plots:
                plt.show()

        if args.mode == 'test':
            if args.model_name == 'base':
                img_size = 100
            elif args.model_name == 'alexnet':
                img_size = 224
            else:
                img_size = 299
            test_dataset, test_loader = load_test_dataset(args.dataset_path, batch_size=args.batch_size, img_size=img_size)
            test_accuracy, conf_matrix, class_accuracy, misclassifications = evaluate_model(model, test_loader, class_names)

    # Plot Seaborn confusion matrix heatmap for better visualization
    plt.figure(figsize=(15, 12))
    sns.heatmap(evaluation_results["conf_matrix"], annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix- {}'.format(model.__class__.__name__))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig("confusion_matrix_{}.png".format(model.__class__.__name__), format="png")
    if args.show_plots:
        plt.show()

    # Plot individual class accuracy using a line plot
    plt.figure(figsize=(15, 8))
    plt.plot(class_names, list(evaluation_results['class_accuracy'].values()), marker='o', linestyle='-', color='blue')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Individual Class Accuracy - {}'.format(model.__class__.__name__))
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)  # Set y-axis limit to represent accuracy as a percentage
    plt.grid(True)
    plt.savefig("class_accuracy_{}.png".format(model.__class__.__name__), format="png")
    if args.show_plots:
        plt.show()

    # Calculate and plot misclassifications heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(evaluation_results["misclassifications"], annot=True, fmt="d", cmap="Reds", xticklabels=class_names, yticklabels=class_names)
    plt.title('Misclassifications Heatmap')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig("misclassifications_{}.png".format(model.__class__.__name__), format="png")
    if args.show_plots:
        plt.show()

    # Plot one sample image of each class with class name and misclassification error as title
    fig, axs = plt.subplots(6, 8, figsize=(16, 12))
    fig.suptitle('Sample Images with Class Name and Misclassification Error', fontsize=16)

    class_labels = evaluation_results["class_labels"]
    class_predictions = evaluation_results["class_predictions"]
    test_dataset,_ = load_test_dataset(args.dataset_path, batch_size=args.batch_size, img_size=img_size)
    misclassification_percentage = {class_name: 100 * np.sum(class_labels[class_labels == i] != class_predictions[class_labels == i]) / np.sum(class_labels == i) for i, class_name in enumerate(test_dataset.classes)}

    for i, ax in enumerate(axs.flatten()):
        class_indices = np.where(class_labels == i)[0]
        if len(class_indices) > 0:
            sample_index = class_indices[0]
            sample_image = (test_dataset[sample_index][0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            sample_class = test_dataset.classes[i]
            title = f'{sample_class}\nError: {misclassification_percentage[sample_class]:.1f}%'
            ax.imshow(sample_image)
            ax.set_title(title)
        ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("classWiseError_{}.png".format(model.__class__.__name__), format="png")
    if args.show_plots:
        plt.show()
