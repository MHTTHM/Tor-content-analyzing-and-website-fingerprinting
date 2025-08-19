import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict, Optional, Any


# --- Assuming RF_Backbone and its helper functions are defined as in the previous response ---
# (make_first_layers, make_1d_feature_stages, RF_Backbone)
# For brevity, I'll paste them here again, slightly condensed.

def make_first_layers(in_channels=1, out_channel_initial=32):
    layers = []
    conv2d1 = nn.Conv2d(in_channels, out_channel_initial, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d1, nn.BatchNorm2d(out_channel_initial, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]
    conv2d2 = nn.Conv2d(out_channel_initial, out_channel_initial, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d2, nn.BatchNorm2d(out_channel_initial, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]
    layers += [nn.MaxPool2d((1, 3)), nn.Dropout(0.1)]
    current_out_channels = 64
    conv2d3 = nn.Conv2d(out_channel_initial, current_out_channels, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d3, nn.BatchNorm2d(current_out_channels, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]
    conv2d4 = nn.Conv2d(current_out_channels, current_out_channels, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d4, nn.BatchNorm2d(current_out_channels, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]
    layers += [nn.MaxPool2d((2, 2)), nn.Dropout(0.1)]
    return nn.Sequential(*layers), current_out_channels

def make_1d_feature_stages(cfg_list, in_channels_1d):
    stages = nn.ModuleList()
    current_channels = in_channels_1d
    stage_layers_collector = []
    for i, v in enumerate(cfg_list):
        if v == 'M':
            stage_layers_collector += [nn.MaxPool1d(3), nn.Dropout(0.3)] # Your config has MaxPool1d(3)
            stages.append(nn.Sequential(*stage_layers_collector))
            stage_layers_collector = []
        else:
            conv1d = nn.Conv1d(current_channels, v, kernel_size=3, stride=1, padding=1)
            stage_layers_collector += [conv1d, nn.BatchNorm1d(v, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]
            current_channels = v
    if stage_layers_collector:
        stages.append(nn.Sequential(*stage_layers_collector))
    return stages

class RF_Backbone(nn.Module):
    def __init__(self, cfg_1d, init_weights=True):
        super(RF_Backbone, self).__init__()
        self.first_layer_input_channels = 1
        self.initial_2d_processor, self.channels_after_2d_processing = make_first_layers(
            in_channels=self.first_layer_input_channels, out_channel_initial=32
        )
        self.feature_stages_1d = make_1d_feature_stages(
            cfg_list=cfg_1d, in_channels_1d=self.channels_after_2d_processing
        )
        if init_weights: self._initialize_weights()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.initial_2d_processor(x)
        x = x.view(x.size(0), self.channels_after_2d_processing, -1)
        outputs = []
        current_features = x
        for stage_module in self.feature_stages_1d:
            current_features = stage_module(current_features)
            outputs.append(current_features)
        return outputs # Outputs are [c3, c4, c5] equivalent, from shallowest to deepest

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else: # Conv1d
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1); m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01); m.bias.data.zero_()

# --- How to use it ---
if __name__ == '__main__':

    '''
##############backbone test###################################################################################
    # Configuration for the 1D feature stages of the backbone
    cfg_1d_backbone = {
        'N': [128, 128, 'M', 256, 256, 'M', 512] # From your original code
    }

    # 1. Instantiate the Backbone
    backbone = RF_Backbone(cfg_1d=cfg_1d_backbone['N'])

    # Create a dummy input tensor
    # Batch size = 2, Input Channels = 2, Sequence Length = 2000
    dummy_input_sequence = torch.randn(2, 2, 2000)

    # 2. Get feature maps from the backbone
    # These are [c3_feat, c4_feat, c5_feat] (example names)
    # Their channel counts are [128, 256, 512] based on cfg_1d_backbone['N']
    backbone_output_features = backbone(dummy_input_sequence)

    print("--- Backbone Outputs ---")
    backbone_out_channels = []
    for i, feat in enumerate(backbone_output_features):
        print(f"Backbone feature {i} shape: {feat.shape}")
        backbone_out_channels.append(feat.shape[1]) # Get channel dimension


##############FPN test###################################################################################
    # 3. Instantiate the FPN
    # The in_channels_list for FPN must match the channel counts from the backbone outputs
    fpn_out_channels_common = 256 # A common choice for FPN output channels
    fpn = FeaturePyramidNetwork1D(
        in_channels_list=backbone_out_channels, # e.g., [128, 256, 512]
        out_channels=fpn_out_channels_common,
        extra_blocks=True # To get P6
    )
    print("FPN shape")
    print(fpn)

    # 4. Get FPN feature maps
    # These will be [p3_feat, p4_feat, p5_feat, p6_feat] (example names)
    # All P-layers will have `fpn_out_channels_common` (e.g., 256) channels.
    fpn_output_features = fpn(backbone_output_features)

    print("\n--- FPN Outputs ---")
    for i, feat in enumerate(fpn_output_features):
        print(f"FPN feature P{i+len(backbone_output_features)-len(fpn_output_features)+2} shape: {feat.shape}")
        # Naming P-layers: if backbone gives 3 levels (C3,C4,C5) and FPN gives 4 (P3,P4,P5,P6)
        # P_idx_start = 3 (assuming C3 is the first level from backbone for FPN)
        # So, fpn_output_features[0] is P3, [1] is P4, [2] is P5, [3] is P6.
        # A simpler way to label for the print:
        # p_level = i + (len(backbone_features) if not fpn.extra_blocks else len(backbone_features) -1) # A bit heuristic for naming
        # Let's just use a generic naming based on the number of backbone levels used for FPN.
        # If backbone_features has 3 levels [c_shallow, c_mid, c_deep]
        # fpn_outputs will be [p_shallow, p_mid, p_deep] and optionally p_deeper (p6)
        # A common notation is P_level where level corresponds to log2(stride).
        # For simplicity, let's assume the first backbone output corresponds to P3.
        actual_p_level = i + (len(backbone_output_features) - (len(fpn_output_features) if not fpn.extra_blocks else len(fpn_output_features)-1) ) +1
        print(f"FPN feature (approx P{i+3}) shape: {feat.shape}")

    '''
        
