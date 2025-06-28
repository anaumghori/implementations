import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import requests
import urllib.request
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# Dataset utilities
def get_class_names():
    url = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
    return [x.replace(' ', '_') for x in requests.get(url).text.splitlines()]

def download_dataset(root="./dataset", limit=15):
    class_names = get_class_names()
    root = Path(root)
    root.mkdir(exist_ok=True, parents=True)
    base_url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
    
    for class_name in tqdm(class_names[:limit], desc="Downloading"):
        file_path = root / f"{class_name}.npy"
        if not file_path.exists():
            url = f"{base_url}{class_name.replace('_', '%20')}.npy"
            urllib.request.urlretrieve(url, file_path)

def load_data(root="./dataset", max_items=10800):
    files = sorted(Path(root).glob('*.npy'))
    samples_per_class = max_items // len(files)
    remainder = max_items % len(files)
    
    x_list, y_list, class_names = [], [], []
    
    for label, file in enumerate(files):
        samples_needed = samples_per_class + (1 if label < remainder else 0)
        data = np.load(file, mmap_mode='r')
        
        if len(data) >= samples_needed:
            indices = np.random.choice(len(data), samples_needed, replace=False)
            selected_data = data[indices]
        else:
            selected_data = data
        
        x_list.append(selected_data)
        y_list.extend([label] * len(selected_data))
        class_names.append(file.stem)
    
    x = np.vstack(x_list)
    y = np.array(y_list, dtype=np.long)
    
    # Shuffle the data
    indices = np.random.permutation(len(x))
    return x[indices], y[indices], class_names

class QuickDrawDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.X = x
        self.Y = y

    def __getitem__(self, idx):
        x = (self.X[idx] / 255.).astype(np.float32).reshape(1, 28, 28)
        y = self.Y[idx]
        return torch.from_numpy(x), int(y)

    def __len__(self):
        return len(self.X)

class Autoencoder(nn.Module):
    def __init__(self, num_features=784, hidden_dim=256):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_features),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, encoded):
        return self.decoder(encoded)
    
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

def save_reconstruction_images(original, reconstructed, epoch, save_dir="./images"):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    
    for i in range(min(8, len(original))):
        axes[0, i].imshow(original[i].reshape(28, 28), cmap='gray')
        axes[0, i].set_title('Original', fontsize=10)
        axes[0, i].axis('off')
        
        axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        axes[1, i].set_title('Reconstructed', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/reconstruction_epoch_{epoch+1}.png", dpi=150, bbox_inches='tight')
    plt.close()

def train_autoencoder():
    # Hyperparameters
    batch_size = 512
    learning_rate = 1e-3
    num_epochs = 50
    hidden_dim = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Download and load data
    download_dataset("./dataset", limit=15)
    x, y, class_names = load_data("./dataset", max_items=12000)
    
    # Split data
    split_idx = int(0.85 * len(x))
    train_x, test_x = x[:split_idx], x[split_idx:]
    train_y, test_y = y[:split_idx], y[split_idx:]
    
    # Create datasets and loaders
    train_dataset = QuickDrawDataset(train_x, train_y)
    test_dataset = QuickDrawDataset(test_x, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize model
    model = Autoencoder(num_features=784, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.MSELoss()
    
    print(f"Training with {len(train_dataset)} samples, testing with {len(test_dataset)} samples")
    print(f"Classes: {class_names}")
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        # Training phase
        model.train()
        train_loss = 0
        num_batches = 0
        
        for features, _ in train_loader:
            features = features.view(-1, 784).to(device)
            
            optimizer.zero_grad()
            reconstructed = model(features)
            loss = criterion(reconstructed, features)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        # Validation phase
        model.eval()
        test_loss = 0
        test_batches = 0
        
        with torch.no_grad():
            for features, _ in test_loader:
                features = features.view(-1, 784).to(device)
                reconstructed = model(features)
                test_loss += criterion(reconstructed, features).item()
                test_batches += 1
                
                # Save reconstruction images for first batch of first few epochs
                if test_batches == 1 and epoch % 10 == 0:
                    save_reconstruction_images(
                        features[:8].cpu().numpy(),
                        reconstructed[:8].cpu().numpy(),
                        epoch
                    )
        
        avg_train_loss = train_loss / num_batches
        avg_test_loss = test_loss / test_batches
        
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}')
    
    print("Training completed")
    
    # Generate final reconstructions
    model.eval()
    
    with torch.no_grad():
        test_features, _ = next(iter(test_loader))
        test_features = test_features.view(-1, 784).to(device)
        final_reconstructions = model(test_features)
        
        save_reconstruction_images(
            test_features[:8].cpu().numpy(),
            final_reconstructions[:8].cpu().numpy(),
            -1
        )
    
    return model, class_names

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    model, class_names = train_autoencoder()
