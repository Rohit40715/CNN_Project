import torch
import torch.nn as nn

class CustomDepthCNN(nn.Module):
    def __init__(self):
        super(CustomDepthCNN, self).__init__()
        
        # CONCEPT: Encoder (Feature Extraction)
        # We use Convolutional layers to slide filters across the image to find patterns.
        self.encoder = nn.Sequential(
            # Input: RGB image (3 channels). Output: 64 feature maps.
            nn.Conv2d(3, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64),
            # CONCEPT: Activation (ReLU) - introduces non-linearity so the model can learn complex relationships.
            nn.ReLU(),                                  
            # CONCEPT: Max Pooling - reduces spatial size (downsampling) while keeping important features.
            nn.MaxPool2d(2, 2),                         
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # CONCEPT: Decoder (Spatial Reconstruction)
        # We use Transposed Convolutions to "un-shrink" abstract features back to image size.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            # Final output: 1 channel representing the Predicted Depth Map.
            nn.Conv2d(32, 1, kernel_size=3, padding=1) 
        )

    def forward(self, x):
        # This defines the "Flow" of data: Image -> Encoder -> Decoder -> Depth Map.
        x = self.encoder(x)
        x = self.decoder(x)
        return x