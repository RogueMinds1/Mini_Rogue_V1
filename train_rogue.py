import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from mini_rogue_v1 import get_model  # Ensure this module is correct and available
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, text, block_size):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.data = [self.stoi[c] for c in text]
        
        # Debugging print statements
        print(f"Text length: {len(text)}")
        print(f"Data length: {len(self.data)}")
        print(f"Block size: {self.block_size}")
        print(f"Number of samples: {self.__len__()}")

    def __len__(self):
        return max(0, len(self.data) - self.block_size + 1)  # Ensure non-negative length

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        if len(chunk) < self.block_size + 1:
            chunk += [0] * (self.block_size + 1 - len(chunk))  # Padding with zeros
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def train_model(text, epochs=10, batch_size=64, lr=1e-4, block_size=128, update_interval=15):
    dataset = TextDataset(text, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)  # Use 2 workers

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(dataset.vocab_size, block_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None  # Mixed precision training scaler
    
    total_start_time = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0  # Initialize total_loss to 0.0
        epoch_start_time = time.time()
        start_time = epoch_start_time
        
        num_batches = len(dataloader)
        with tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for i, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                if scaler:
                    with torch.cuda.amp.autocast():  # Enable mixed precision training
                        logits, loss = model(x, y)
                    scaler.scale(loss).backward()  # Scale the loss for mixed precision
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits, loss = model(x, y)
                    loss.backward()
                    optimizer.step()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping

                batch_loss = loss.item()
                total_loss += batch_loss  # Accumulate batch loss
                
                current_time = time.time()
                elapsed_time = current_time - total_start_time
                avg_time_per_batch = elapsed_time / ((epoch * num_batches) + (i + 1))
                remaining_batches = (epochs * num_batches) - ((epoch * num_batches) + (i + 1))
                eta = remaining_batches * avg_time_per_batch
                
                pbar.set_postfix({"Loss": total_loss/(i+1), "ETA": f"{int(eta // 3600):02}:{int((eta % 3600) // 60):02}:{int(eta % 60):02}"})
                pbar.update(1)
                
                if current_time - start_time > update_interval:
                    print(f'Epoch {epoch+1}, Batch {i+1}/{num_batches}, Loss: {total_loss/(i+1)}')
                    start_time = current_time
        
        epoch_duration = time.time() - epoch_start_time
        print(f'Epoch {epoch+1} completed, Average Loss: {total_loss/num_batches}, Duration: {epoch_duration:.2f} seconds')
    
    # Save the trained model outside the loop
    torch.save(model.state_dict(), 'mini_rogue_model.pth')

if __name__ == "__main__":
    with open(r"C:\RMI-CODE\Models\Mini_Rogue_V1\Mini_Rogue_V1\data\cleaned_text.txt", "r", encoding="utf-8") as f:
        text = f.read()
    train_model(text)
