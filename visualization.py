import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Function to load images and labels
def load_dataset(folder_path):
    data = []
    labels = []
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        for image_file in os.listdir(label_path):
            image_path = os.path.join(label_path, image_file)
            data.append(image_path)
            labels.append(label)
    return pd.DataFrame({'Image_Path': data, 'Label': labels})

# Create a folder to save plots
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load the dataset from the folder
dataset_path = r'fruitData/Training'
df = load_dataset(dataset_path)

# Display basic information about the dataset
print("Number of classes:", df['Label'].nunique())
print("Total images in Training dataset:", len(df))

# Visualize class distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='Label', data=df)
plt.title('Class Distribution-Training Dataset')
plt.xticks(rotation=90)
plt.savefig('plots/class_distribution_training.png')
plt.show()

# Display sample images for each class
fig, axes = plt.subplots(4, 11, figsize=(20, 8))
for i, (label, group) in enumerate(df.groupby('Label')):
    ax = axes[i // 11, i % 11]
    image_path = group['Image_Path'].iloc[0]
    image = Image.open(image_path)
    ax.imshow(image)
    ax.axis('off')
    ax.set_title(label)
plt.savefig('plots/sample_images_training.png')
plt.show()

# Check for outliers based on the number of images per class
plt.figure(figsize=(12, 6))
sns.boxplot(x='Label', y=df.groupby('Label')['Image_Path'].transform('count'), data=df)
plt.title('Outlier Detection')
plt.xticks(rotation=90)
plt.savefig('plots/outlier_detection.png')
plt.show()

### Testing Dataset

# Load the dataset from the folder
dataset_path = r'fruitData/Test'
df_test = load_dataset(dataset_path)

# Display basic information about the dataset
print("Number of classes:", df_test['Label'].nunique())
print("Total images in Testing dataset:", len(df_test))

# Visualize class distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='Label', data=df_test)
plt.title('Class Distribution-Testing Dataset')
plt.xticks(rotation=90)
plt.savefig('plots/class_distribution_testing.png')
plt.show()

# Display sample images for each class
fig, axes = plt.subplots(4, 11, figsize=(20, 8))
for i, (label, group) in enumerate(df_test.groupby('Label')):
    ax = axes[i // 11, i % 11]
    image_path = group['Image_Path'].iloc[0]
    image = Image.open(image_path)
    ax.imshow(image)
    ax.axis('off')
    ax.set_title(label)
plt.savefig('plots/sample_images_testing.png')
plt.show()
