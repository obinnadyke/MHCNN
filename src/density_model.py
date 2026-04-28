"""
density_model.py - Deep density model with multi-task outputs |(c) itrustal.com
Provisions:
1. DensityModel: Base + Regressor + Classifier with deeper network
2. setup_model: Creates the model and wraps with DataParallel if multiple GPUs exist
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from device_checker import move_to_device, verify_device_placement

class DeepUNetDecoder(nn.Module):
    """
    Deep UNet decoder with more skip connections and channels
    """
    def __init__(self, encoder_channels, decoder_channels=(256, 128, 64, 32, 16)):
        super().__init__()

        # Get encoder channels information
        self.encoder_channels = encoder_channels

        # Initialize decoder blocks list
        self.blocks = nn.ModuleList()

        # Initial encoder features (output of deepest encoder block)
        encoder_features = encoder_channels[-1]

        # Create decoder blocks
        for i, decoder_features in enumerate(decoder_channels):
            # For first block, no encoder connection (bottleneck)
            if i == 0:
                self.blocks.append(
                    DecoderBlock(
                        encoder_features,
                        decoder_features,
                        skip_channels=0  # No skip for bottleneck
                    )
                )
            else:
                # Connect to encoder with skip connections
                # Use encoder features from the corresponding level
                encoder_idx = len(encoder_channels) - i - 1
                if encoder_idx >= 0:  # Make sure we don't go out of bounds
                    skip_features = encoder_channels[encoder_idx]
                else:
                    skip_features = 0

                self.blocks.append(
                    DecoderBlock(
                        decoder_channels[i-1],
                        decoder_features,
                        skip_channels=skip_features
                    )
                )

        # Final conv layer to get to output classes
        self.final_conv = nn.Conv2d(decoder_channels[-1], 1, kernel_size=1)

        self.activation = nn.Sigmoid()

    def forward(self, features):
        """
        Forward pass through decoder
        Args - features: List of features from encoder blocks
        """
        # Reverse features from encoder to use them in correct order
        features = features[::-1]

        # Get the deep features (output of deep encoder block)
        x = features[0]

        # Go through decoder blocks
        skips = features[1:]  # Skip connections from encoder

        for i, decoder_block in enumerate(self.blocks):
            if i == 0:  # First block - no skip connection
                x = decoder_block(x, None)
            elif i <= len(skips):  # Blocks with skip connections
                x = decoder_block(x, skips[i-1])
            else:  # Extra blocks without skip connections
                x = decoder_block(x, None)

        # Final conv and activation
        x = self.final_conv(x)
        return self.activation(x)


class DecoderBlock(nn.Module):
    """
    Custom decoder block with improved upsampling and feature fusion
    """
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()

        self.has_skip = skip_channels > 0

        # First, we need to upsample the features
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Then, we process the upsampled features
        # If there's a skip connection, we add the skip channels
        conv_in_channels = in_channels + (skip_channels if self.has_skip else 0)

        # Two conv blocks with batch norm and activation
        self.conv1 = nn.Sequential(
            nn.Conv2d(conv_in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True)
        )

        # Spatial attention for feature integration
        self.attention = SpatialAttention(out_channels)

    def forward(self, x, skip=None):
        x = self.upsample(x)

        if self.has_skip and skip is not None:
            # Handle different spatial sizes
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)

            # Concatenate along channel dimension
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)

        return x


class SpatialAttention(nn.Module):
    """
    Spatial attention module for better feature focus
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Generate attention map
        attention = self.sigmoid(self.conv(x))

        # Apply attention
        return x * attention


class EnhancedDensityModel(nn.Module):
    def __init__(self, base_model, output_size=512, use_deep_decoder=True):
        super().__init__()
        self.output_size = output_size

        # Option to use custom deep decoder for better segmentation
        self.use_deep_decoder = use_deep_decoder

        # If using deep decoder, we only need the encoder from base model
        if use_deep_decoder:
            # Extract encoder from base model
            if hasattr(base_model, 'encoder'):
                self.encoder = base_model.encoder
            elif hasattr(base_model, 'module') and hasattr(base_model.module, 'encoder'):
                self.encoder = base_model.module.encoder
            else:
                raise ValueError("Could not extract encoder from base model")
        else:
            # Use the entire base model as-is
            self.seg_model = base_model

        # The base model is already on some device
        if use_deep_decoder:
            self.device = next(self.encoder.parameters()).device
        else:
            self.device = next(base_model.parameters()).device

        print(f"[EnhancedDensityModel] Initialized on device: {self.device}")

        # Detect encoder channels
        encoder_channels = self._get_encoder_channels()
        print(f"[EnhancedDensityModel] Encoder channels: {encoder_channels}")

        # Create custom deep decoder if requested
        if use_deep_decoder:
            self.decoder = DeepUNetDecoder(encoder_channels).to(self.device)

        # Deep regression branch with attention
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_channels[-1], 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.6),
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(self.device)

        # initialize regressor for better training stability
        with torch.no_grad():
            # Initialize final layer biases
            self.regressor[-2].bias.fill_(0.5)  # Middle value for sigmoid

        # Deep classification branch
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder_channels[-1], 512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 4)  # 4 BI-RADS classes (1-4)
        ).to(self.device)

        # Initialize classification head with balanced bias for stable training
        with torch.no_grad():
            if hasattr(self.classifier[-1], 'bias'):
                self.classifier[-1].bias.fill_(0)  # Start with equal probability for all classes

        # Initialize weights with Xavier uniform for better convergence
        for m in self.regressor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None and m is not self.regressor[-2]:  # Skip the final layer which we already initialized
                    nn.init.constant_(m.bias, 0)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None and m is not self.classifier[-1]:  # Skip the final layer which we already initialized
                    nn.init.constant_(m.bias, 0)

        # Double-check everything is on the same device
        self.to(self.device)
        verify_device_placement(self, self.device)

    def forward(self, x):
        """
        Forward pass with full gradient flow through the encoder 
        Args - x (torch.Tensor): Input tensor of shape [B, 3, H, W]
        Returns - tuple: (segmentation_output, regression_output, classification_output)
        """
        if self.use_deep_decoder:
            # Forward through encoder
            features = self.encoder(x)

            # Forward through decoder
            seg_out = self.decoder(features)
        else:
            # Forward through segmentation model
            seg_out = self.seg_model(x)

            # Get encoder from base model
            encoder = self.seg_model.module.encoder if hasattr(self.seg_model, 'module') \
                    else self.seg_model.encoder

            # Forward through encoder to get features
            features = encoder(x)

        # Get the deepest features for regression and classification
        deep_features = features[-1]

        # Forward through regression branch
        reg_out = self.regressor(deep_features)

        # Forward through classification branch
        cls_out = self.classifier(deep_features)

        return seg_out, reg_out, cls_out

    def _get_encoder_channels(self):
        try:
            dummy = torch.zeros(1, 3, self.output_size, self.output_size, device=self.device)

            if self.use_deep_decoder:
                encoder = self.encoder
            else:
                encoder = self.seg_model.module.encoder if hasattr(self.seg_model, 'module') \
                        else self.seg_model.encoder

            with torch.no_grad():
                feats = encoder(dummy)

            channels = [f.shape[1] for f in feats]
            return channels
        except Exception as e:
            print(f"[DensityModel] WARNING: Could not extract encoder channels. Using default. Error: {e}")
            return [64, 128, 256, 512, 2048]  # Default for ResNet

def setup_model(config, device_manager):
    """
    Create the base segmentation model from segmentation_models_multi_tasking,
    then wrap it inside DensityModel.
    """
    import segmentation_models_multi_tasking as smp

    device = device_manager.device

    # Option to use deep decoder
    use_deep_decoder = config.get('use_deep_decoder', True)
    print(f"[setup_model] Using deep custom decoder: {use_deep_decoder}")

    # 1) Create base segmentation model
    base_model = getattr(smp, config['segmentation_model'])(
        encoder_name=config['encoder'],
        encoder_weights=config.get('pretrained_weights', 'imagenet'),
        classes=1,
        activation=config.get('activation_function', 'sigmoid'),
        # Parameters for deeper model
        decoder_channels=(256, 128, 64, 32, 16),  # Deeper decoder
        decoder_use_batchnorm=True
    ).to(device)

    # 2) Wrap in DensityModel
    model = EnhancedDensityModel(
        base_model,
        output_size=config.get('output_size', 512),
        use_deep_decoder=use_deep_decoder
    )

    # 3) DataParallel if multiple GPUs
    model = device_manager.setup_dataparallel(model)

    # 4) Final check
    verify_device_placement(model, device)
    return model

def load_model_for_inference(model_path, config, device):
    """
    Loads a trained model checkpoint for inference.
    """
    import segmentation_models_multi_tasking as smp

    # Get deep decoder option
    use_deep_decoder = config.get('use_deep_decoder', True)
    print(f"[load_model_for_inference] Using deep custom decoder: {use_deep_decoder}")

    # Create base seg model
    base_model = getattr(smp, config['segmentation_model'])(
        encoder_name=config['encoder'],
        encoder_weights=None,  # Don't load pretrained weights for inference
        classes=1,
        activation='sigmoid',
        # parameters for deeper model
        decoder_channels=(256, 128, 64, 32, 16),
        decoder_use_batchnorm=True
    )
    base_model.to(device)

    # Create EnhancedDensityModel
    model = EnhancedDensityModel(
        base_model,
        output_size=config.get('output_size', 512),
        use_deep_decoder=use_deep_decoder
    )
    model.to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # If DataParallel was used, remove 'module.' prefix
    new_dict = {}
    for k, v in state_dict.items():
        new_k = k[7:] if k.startswith('module.') else k
        new_dict[new_k] = v

    # Error handling for model loading
    try:
        model.load_state_dict(new_dict, strict=False)
        print(f"[load_model_for_inference] Loaded model from {model_path}")
    except Exception as e:
        print(f"[load_model_for_inference] WARNING: Could not load state dict with strict=False: {e}")
        print("[load_model_for_inference] Trying to load with custom key matching...")

        # Try matching keys by name pattern
        matched_dict = {}
        model_dict = model.state_dict()

        for model_key in model_dict.keys():
            for checkpoint_key, value in new_dict.items():
                if model_key in checkpoint_key or checkpoint_key in model_key:
                    if model_dict[model_key].shape == value.shape:
                        matched_dict[model_key] = value
                        break

        # Load matched keys
        if matched_dict:
            model.load_state_dict(matched_dict, strict=False)
            print(f"[load_model_for_inference] Loaded {len(matched_dict)}/{len(model_dict)} model parameters")
        else:
            print("[load_model_for_inference] Could not match any keys. Using random initialization.")

    model.eval()
    return model
