import torch
import torch.nn as nn
from model.backbone import DiffTransformerEncoder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Test code
def test_diff_transformer_encoder():
    # Create a batch of RGB images with size [batch_size, channels, height, width]
    batch_size = 4  # You can modify the batch size as needed
    height, width = 256, 256  # You can modify the height and width of the image
    x = torch.randn(batch_size, 3, height, width)  # Simulating a batch of RGB images
    x = x.to(device)

    # Instantiate the DiffTransformerEncoder model
    model = DiffTransformerEncoder(in_chans=3).to(device)

    # Forward pass through the model
    feature_maps = model(x)

    # Print the shapes of the input and the feature maps at each stage
    print("Input shape:", x.shape)
    for i, feature_map in enumerate(feature_maps):
        print(f"Feature map at stage {i} shape: {feature_map.shape}")

# Run the test
test_diff_transformer_encoder()
