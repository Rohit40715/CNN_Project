import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.custom_cnn import CustomDepthCNN # Or your Unet
from utils import KITTIDataset

# --- 1. Hyperparameters [cite: 232] ---
BATCH_SIZE = 4      # Optimized for 16GB RAM and high-res images
LEARNING_RATE = 1e-4
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGES_ROOT = os.getenv("KITTI_IMAGES_ROOT", "dataset/images")
DEPTHS_ROOT = os.getenv("KITTI_DEPTHS_ROOT", "dataset/depths")
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))

def main():
    # --- 2. Data & Model Setup ---
    dataset = KITTIDataset(images_root=IMAGES_ROOT, depths_root=DEPTHS_ROOT)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    model = CustomDepthCNN().to(DEVICE)
    # [cite: 173, 230] Adam Optimizer: Mathematically updates weights to minimize error
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3. Training Loop ---
    print(f"Starting Training on {DEVICE}...")

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for batch_idx, (images, depths, masks, _, _) in enumerate(train_loader):
            images, depths, masks = images.to(DEVICE), depths.to(DEVICE), masks.to(DEVICE)

            # A. Forward Pass [cite: 170]
            preds = model(images)

            # B. Masked MSE Loss Calculation [cite: 168, 297, 343]
            # We calculate (Pred - Actual)^2 then multiply by mask to ignore black pixels
            diff = (preds - depths) ** 2
            masked_diff = diff * masks
            # [cite: 339, 343] Average the loss only over the 'Valid' pixels
            loss = masked_diff.sum() / (masks.sum() + 1e-6)

            # C. Backpropagation [cite: 171, 172, 229]
            optimizer.zero_grad() # Clear old gradients
            loss.backward()      # Calculate new gradients using calculus
            optimizer.step()      # Update model weights

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Average Loss: {epoch_loss/len(train_loader):.4f}")

        # [cite: 111] Save checkpoint every epoch
        torch.save(model.state_dict(), f"checkpoints/depth_model_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    main()