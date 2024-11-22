import torch
import torch.nn as nn
from torchinfo import summary
from model.backbone import DiffTransformerEncoder

torch.cuda.empty_cache()  # 释放未使用的 GPU 内存
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Test code
def test_diff_transformer_encoder():
    # Create a batch of RGB images with size [batch_size, channels, height, width]
    batch_size = 1
    height, width = 512, 512
    x = torch.randn(batch_size, 3, height, width)
    x = x.to(device)

    model = DiffTransformerEncoder(in_chans=3).to(device)
    model.eval()
    # Use torchinfo to summarize the model
    print("Model Summary:")
    summary(
        model,
        input_size= (1, 3, 512, 512),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        # depth=3  # Control the depth of details in the output
    )
    with torch.no_grad():
        feature_maps = model(x)

    print("Input shape:", x.shape)
    for i, feature_map in enumerate(feature_maps):
        print(f"Feature map at stage {i} shape: {feature_map.shape}")

# Run the test
test_diff_transformer_encoder()
