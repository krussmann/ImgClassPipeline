import torch
import torchvision.models
from data import CustomDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import matplotlib.pyplot as plt
from model import ResNet
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, multilabel_confusion_matrix, classification_report, accuracy_score
import os
import torch.optim as optim
import onnx
from torchvision.models import resnet50, inception_v3


# Define the root directory
root_dir = 'Blood_cell_Cancer'

# Initialize lists to store file paths and labels
file_paths = []
labels = []

# Iterate over subfolders
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        # Append file path
        file_path = os.path.join(subdir, file)
        file_paths.append(file_path)

        # Append label (extracted from the subfolder name)
        label = os.path.basename(subdir)
        labels.append(label)

# Create DataFrame
df = pd.DataFrame({'filename': file_paths, 'label': labels})
df.head()

df.info()
print()
print(df["label"].value_counts())

# One-hot encode the labels
labels_encoded = pd.get_dummies(df['label'])

# Concatenate the one-hot encoded labels DataFrame with the original DataFrame
df_one_hot = df[['filename']].join(labels_encoded)

# Display the modified DataFrame
df_one_hot.head()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Code is running on {device}.")

# Load the data from the DataFrame and perform a train-test-split
train, val_test = train_test_split(df_one_hot, test_size=0.3)
val, test = train_test_split(val_test, test_size=0.5)

# Set up data loading for the training, validation, and test sets
train_data = CustomDataset(train, 'training')
val_data = CustomDataset(val, 'validation')
test_data = CustomDataset(test, 'test')

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)  # No need to shuffle for test

# Create an instance of our ResNet model
model = ResNet()
print(model)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0006)

trainer = Trainer(model=model,
                  crit=criterion,
                  optim=optimizer,
                  train_dl=train_loader,
                  val_test_dl=val_loader,
                  cuda=True,
                  early_stopping_patience=30)

# Call fit on trainer
epochs = 10
train_losses, val_losses = trainer.fit(epochs)

# Plot the training and validation losses
plt.plot(np.arange(len(train_losses)), train_losses, label='train loss')
plt.plot(np.arange(len(val_losses)), val_losses, label='val loss')
plt.legend()
plt.savefig('losses.png')

# Evaluate the model on the test set
total_loss = 0.0
model.eval()
test_targets = []
test_predictions = []
with torch.no_grad():
    for inputs, targets in test_loader:
#         inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        test_targets.extend(targets.cpu().detach().numpy())
        test_predictions.extend(outputs.cpu().detach().numpy())

test_loss = total_loss /len(test_loader.dataset)
test_metric = accuracy_score(np.argmax(np.array(test_targets),axis=1), np.argmax(np.array(test_predictions), axis=1))
print(f"Test Loss: {test_loss:.6f}")
print(f"Accuracy: {test_metric:.4f}")