import os
import pickle
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from models.custom_cnn import CustomDepthCNN  # Or your Unet
from utils import KITTIDataset

# --- 1. Hyperparameters ---
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGES_ROOT = os.getenv("KITTI_IMAGES_ROOT", "dataset/images")
DEPTHS_ROOT = os.getenv("KITTI_DEPTHS_ROOT", "dataset/depths")
VAL_IMAGES_ROOT = os.getenv("KITTI_VAL_IMAGES_ROOT", "dataset_validation/images")
VAL_DEPTHS_ROOT = os.getenv("KITTI_VAL_DEPTHS_ROOT", "dataset_validation/depths")
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))

def main():
    # --- 2. Data & Model Setup ---
    dataset = KITTIDataset(images_root=IMAGES_ROOT, depths_root=DEPTHS_ROOT)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    val_loader = None
    try:
        val_dataset = KITTIDataset(images_root=VAL_IMAGES_ROOT, depths_root=VAL_DEPTHS_ROOT)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        print(f"Loaded validation dataset with {len(val_dataset)} samples.")
    except RuntimeError:
        print(f"No validation data found at {VAL_IMAGES_ROOT} / {VAL_DEPTHS_ROOT}. Skipping validation.")

    model = CustomDepthCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3. Training Loop ---
    print(f"Starting Training on {DEVICE}...")
    os.makedirs("checkpoints", exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, depths, masks, _, _) in enumerate(train_loader):
            images, depths, masks = images.to(DEVICE), depths.to(DEVICE), masks.to(DEVICE)

            preds = model(images)
            if preds.shape[-2:] != depths.shape[-2:]:
                preds = F.interpolate(preds, size=depths.shape[-2:], mode="bilinear", align_corners=False)

            diff = (preds - depths) ** 2
            masked_diff = diff * masks
            loss = masked_diff.sum() / (masks.sum() + 1e-6)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} batch {batch_idx+1}/{len(train_loader)} loss={loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Average Train Loss: {avg_train_loss:.4f}")
        torch.save(model.state_dict(), f"checkpoints/depth_model_epoch_{epoch+1}.pth")

        if val_loader is not None:
            model.eval()
            val_loss_total = 0.0
            valid_pixel_count = 0.0

            with torch.no_grad():
                for images, depths, masks, _, _ in val_loader:
                    images, depths, masks = images.to(DEVICE), depths.to(DEVICE), masks.to(DEVICE)

                    preds = model(images)
                    if preds.shape[-2:] != depths.shape[-2:]:
                        preds = F.interpolate(preds, size=depths.shape[-2:], mode="bilinear", align_corners=False)

                    diff = (preds - depths) ** 2
                    masked_diff = diff * masks
                    val_loss_total += masked_diff.sum().item()
                    valid_pixel_count += masks.sum().item()

            val_loss = val_loss_total / (valid_pixel_count + 1e-6)
            print(f"Epoch [{epoch+1}/{EPOCHS}] - Validation Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "checkpoints/depth_model_best.pth")
                print("Saved best validation checkpoint: checkpoints/depth_model_best.pth")

    # Save the final model as a pickle file
    final_pickle_path = "checkpoints/depth_model_final.pkl"
    model_cpu = model.to(torch.device("cpu"))
    with open(final_pickle_path, "wb") as f:
        pickle.dump(model_cpu, f)

    print(f"Final model saved as pickle file: {final_pickle_path}")

if __name__ == "__main__":
    main()