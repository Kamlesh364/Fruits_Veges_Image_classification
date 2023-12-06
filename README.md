# Fruits and Vegetables Packaging Automation using Computer Vision
In this project, the aim is to automate the packaging process of fruits and vegetables using computer vision. The goal is to develop a robust image classification model that can accurately identify and classify different fruits and vegetables. This model will be integrated into an automated packaging line, ensuring that each item is correctly sorted and packed into the corresponding bundle for sale or further processing. This repository presents information and `Python` code related to the following-

```
1. Dataset
2. Dataset understanding and Visualization
3. Dataset pre-processing and preparation
4. Training methodology and pipelines
5. Trained models and their weights
6. Inference on saved weights
7. Performance analysis
```

# Train and/or Evaluate available models:

## Train.py

This script (`train.py`) is designed for training and evaluating deep learning models on a given dataset. The script supports various models and training configurations. Below is an example command to run the script:

```
python train.py --dataset_path cat_dog_dataset --mode train --model_name custom_mobilenet --epochs 20 --batch_size 64 --split_ratio 0.7 --lr 0.01 --patience 10 --evaluate True
```

## Usage

To run this script, you need to provide the following arguments:

- `--dataset_path`: The path to the folder that contains the `Training` and `Test` folders. The folder should have subfolders for each class of images. For example, if the dataset is for Apple and Grapes classification, the folder structure should be:

```
dataset_path
├── Training
    ├── Apple
    │   ├── Apple1.jpg
    │   ├── Apple2.jpg
    │   └── ...
    └── Grapes
        ├── Grapes1.jpg
        ├── Grapes2.jpg
        └── ...
├── Test
    ├── Apple
    │   ├── Apple1.jpg
    │   ├── Apple2.jpg
    │   └── ...
    └── Grapes
        ├── Grapes1.jpg
        ├── Grapes2.jpg
        └── ...
    ```
```

- `--mode`: The mode of the script, either 'train' or 'test'. If 'train', the script will train a model on the dataset and plot the losses and accuracies. If 'test', the script will load a pretrained model and evaluate it on the dataset.

- `--model_name`: The name of the model to use, either 'base', 'alexnet', 'inception', 'mobilenet', or 'custom_mobilenet'. The 'base' model is a simple convolutional neural network with two convolutional layers and two fully connected layers. The 'alexnet' model is the AlexNet architecture proposed by Krizhevsky et al. in 2012. The 'inception' model is the Inception-v3 architecture proposed by Szegedy et al. in 2015.

- `--epochs`: The number of epochs to train the model for. An epoch is one complete pass through the entire dataset. The default value is 10.

- `--batch_size`: The batch size to use for training and testing. A batch is a subset of the dataset that is fed to the model at once. The default value is 32.

- `--split_ratio`: The ratio of the dataset to use for training and validation. The validation set is used to monitor the performance of the model during training and avoid overfitting. The default value is 0.8, which means 80% of the dataset will be used for training and 20% for validation.

- `--shuffle`: A boolean flag that indicates whether to shuffle the dataset before splitting it into training and validation sets. Shuffling the dataset helps to reduce the bias and variance of the model. The default value is True.

- `--lr`: The learning rate to use for training the model. The learning rate is a hyperparameter that controls how much the model weights are updated in each iteration. A high learning rate can lead to faster convergence, but also to instability and divergence. A low learning rate can lead to slower convergence, but also to better accuracy and generalization. The default value is 0.001.

- `--patience`: The patience to use for early stopping. Early stopping is a technique that stops the training process when the validation loss stops improving for a certain number of epochs. This helps to prevent overfitting and save computational resources. The patience is the number of epochs to wait before stopping the training if the validation loss does not improve. The default value is 5.

- `--evaluate`: A boolean flag that indicates whether to evaluate the model on the test set after training. The test set is a separate set of images that is not used for training or validation, and is used to measure the final performance of the model. The default value is False.

- `--show_plots`: A boolean flag that indicates whether to show the plots of the losses and accuracies after training. The plots are also saved as PNG files in the same folder as the script. The default value is False.

## Example

To train the alexnet model on the cat and dog dataset for 20 epochs, with a batch size of 64, a split ratio of 0.7, a learning rate of 0.01, a patience of 10, and evaluate it on the test set, you can run the following command:

```
python train.py --dataset_path cat_dog_dataset --mode train --model_name alexnet --epochs 20 --batch_size 64 --split_ratio 0.7 --lr 0.01 --patience 10 --evaluate True
```

This will output the training and validation losses and accuracies for each epoch, and the test accuracy at the end. It will also save the trained model as 'alexnet.pt' and the plots as 'AlexNet.png' in the same folder as the script.

## References

: Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).

: Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2818-2826).