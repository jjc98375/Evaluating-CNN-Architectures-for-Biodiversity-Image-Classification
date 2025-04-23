from wildlife_datasets import datasets, loader
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split

# ---- CONFIG ----
# Only use CowDataset and DogFaceNet for now
DATASET_CLASSES = [
    datasets.DogFaceNet,
    datasets.LionData,
    datasets.MacaqueFaces
]

# ---- LOAD AND MERGE DATA ----
import pandas as pd

def load_and_merge(classes):
    dfs = []
    for cls in classes:
        try:
            d = loader.load_dataset(cls, 'data', 'dataframes')
            df = d.df.copy()
            df['dataset'] = cls.__name__
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {cls.__name__}: {e}")
    if not dfs:
        raise RuntimeError("No datasets could be loaded. Please check your data folders.")
    return pd.concat(dfs, ignore_index=True)

merged_df = load_and_merge(DATASET_CLASSES)

# ---- LABEL ENCODING ----
merged_df['identity'] = merged_df['identity'].astype(str)
label2idx = {label: idx for idx, label in enumerate(sorted(merged_df['identity'].unique()))}
merged_df['label'] = merged_df['identity'].map(label2idx)

# ---- DATASET ----
class AnimalDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Build correct image path for each dataset

        if row['dataset'].lower() == 'dogfacenet':
            img_path = os.path.join('data', 'DogFaceNet', row['path'])
        elif row['dataset'].lower() == 'liondata':
            img_path = os.path.join('data', 'LionData', row['path'])
        elif row['dataset'].lower() == 'macaquefaces':
            img_path = os.path.join('data', 'MacaqueFaces', row['path'])
        else:
            raise ValueError(f"Unknown dataset: {row['dataset']}")

        image = Image.open(img_path).convert('RGB')
        label = row['label']
        if self.transform:
            image = self.transform(image)
        return image, label

# ---- SPLIT ----
def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    return train_df, test_df

# ---- MODEL ----
from torchvision.models import Inception_V3_Weights

def get_inception(num_classes):
    # Use the recommended weights argument and aux_logits=True for pretrained
    model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    # Replace the classifier head (trainable)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    # Optionally, disable the auxiliary classifier head after loading weights
    model.aux_logits = False
    return model

if __name__ == "__main__":
    merged_df = load_and_merge(DATASET_CLASSES)

    # ---- LABEL ENCODING ----
    merged_df['identity'] = merged_df['identity'].astype(str)
    label2idx = {label: idx for idx, label in enumerate(sorted(merged_df['identity'].unique()))}
    merged_df['label'] = merged_df['identity'].map(label2idx)

    # ---- SPLIT ----
    train_df, test_df = train_test_split(merged_df, test_size=0.2, stratify=merged_df['label'], random_state=42)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    train_dataset = AnimalDataset(train_df, transform=transform)
    test_dataset = AnimalDataset(test_df, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

    num_classes = len(label2idx)
    model = get_inception(num_classes)

    # ---- TRAINING AND EVAL SCAFFOLD ----
    # Add your training and evaluation loop here
    print('Model and data ready for training!')

    # ---- TRAINING LOOP ----
    import matplotlib.pyplot as plt
    import random
    from tqdm import tqdm
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4)
    import sys
    # Allow user to specify epochs as a third argument
    if len(sys.argv) > 1:
        try:
            epochs = int(sys.argv[1])
        except Exception:
            print('Invalid epoch argument, using default.')
            epochs = 10
    else:
        epochs = 10
    print(f'Starting training for {epochs} epochs...')
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in train_iter:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, tuple):  # Inception returns (main, aux)
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            train_iter.set_postfix(loss=loss.item())
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    print('Training complete!')

    # ---- SAVE TRAINED MODEL ----
    torch.save(model.state_dict(), 'inception_model.pth')
    print('Model saved to inception_model.pth')
    
    
    
    
    

    # # ---- BATCH INSPECTION FOR DATA VERIFICATION ----
    # import matplotlib.pyplot as plt
    # import torchvision

    # dataiter = iter(train_loader)
    # images, labels = next(dataiter)

    # print("Batch image tensor shape:", images.shape)
    # print("Batch label tensor shape:", labels.shape)
    # print("First 5 labels:", labels[:5])

    # # Visualize the first 4 images in the batch
    # img_grid = torchvision.utils.make_grid(images[:4], nrow=4)
    # plt.figure(figsize=(12, 3))
    # plt.imshow(img_grid.permute(1, 2, 0).numpy())
    # plt.title("Sample images from train_loader")
    # plt.axis('off')
    # plt.show()

    # # ---- SHOW FIRST 5 IMAGES AND LABELS FOR COW AND DOG ----
    # import numpy as np
    # # Map integer label back to identity string
    # idx2label = {v: k for k, v in label2idx.items()}

    # def show_samples_by_class(df, class_name, dataset_name, n=5):
    #     # Filter by dataset and class
    #     filtered = df[(df['dataset'] == dataset_name) & (df['identity'] == class_name)]
    #     samples = filtered.head(n)
    #     images, labels = [], []
    #     for _, row in samples.iterrows():
    #         if row['dataset'].lower() == 'cowdataset':
    #             img_path = os.path.join('data', 'CowDataset', row['path'])
    #         elif row['dataset'].lower() == 'dogfacenet':
    #             img_path = os.path.join('data', 'DogFaceNet', row['path'])
    #         else:
    #             continue
    #         img = Image.open(img_path).convert('RGB').resize((128, 128))
    #         images.append(np.asarray(img))
    #         labels.append(row['identity'])
    #     return images, labels

    # # Pick the first available identity for each dataset as an example
    # cow_id = merged_df[merged_df['dataset'] == 'CowDataset']['identity'].iloc[0]
    # dog_id = merged_df[merged_df['dataset'] == 'DogFaceNet']['identity'].iloc[0]

    # cow_imgs, cow_labels = show_samples_by_class(merged_df, cow_id, 'CowDataset', n=5)
    # dog_imgs, dog_labels = show_samples_by_class(merged_df, dog_id, 'DogFaceNet', n=5)

    # fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    # for i in range(5):
    #     axes[0, i].imshow(cow_imgs[i])
    #     axes[0, i].set_title(f"Cow: {cow_labels[i]}")
    #     axes[0, i].axis('off')
    #     axes[1, i].imshow(dog_imgs[i])
    #     axes[1, i].set_title(f"Dog: {dog_labels[i]}")
    #     axes[1, i].axis('off')
    # plt.suptitle("First 5 Images and Labels for Cow and Dog")
    # plt.tight_layout()
    # plt.show()