import torch
import torch.nn as nn
import torch.nn.functional as F
import math, os, sys
from typing import List, Tuple, Dict, Optional, Any

current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录（utils）
parent_dir = os.path.dirname(current_dir)  # 项目根目录
sys.path.append(parent_dir)
multi_dir = os.path.join(parent_dir, "multi-tab")
sys.path.append(multi_dir)
# extractor_dir = os.path.join(parent_dir, "models")
# sys.path.append(extractor_dir)

from multitab_func import DetectionLoss

# --- 1D Feature Pyramid Network (FPN) ---
class FeaturePyramidNetwork1D(nn.Module):
    def __init__(self, in_channels_list, out_channels, extra_blocks=True):
        """
        Args:
            in_channels_list (list[int]): List of anumber of channels for each input feature map
                                          from the backbone, e.g., [c3_channels, c4_channels, c5_channels].
                                          Assumed to be ordered from shallower to deeper.
            out_channels (int): Number of channels for all output FPN feature maps (P-layers).
            extra_blocks (bool): If True, add P6 (and P7 if using RetinaNetP6P7) by downsampling
                                 the deepest P-layer. For now, just P6.
        """
        super(FeaturePyramidNetwork1D, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.extra_blocks = extra_blocks

        # Lateral connections (1x1 conv to project backbone features to FPN out_channels)
        self.lateral_convs = nn.ModuleList()
        # Output convolutions (3x3 conv for final P-layers)
        self.fpn_convs = nn.ModuleList()

        for in_channels in self.in_channels_list:
            self.lateral_convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
            self.fpn_convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1))

        # Extra P6 layer if needed
        if self.extra_blocks:
            # P6 is typically made from P5 (or C5) by max pooling or strided convolution
            # Using MaxPool1d for simplicity, similar to some implementations
            self.p6_conv = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            # Or you could use MaxPool1d: self.p6_pool = nn.MaxPool1d(kernel_size=1, stride=2) if P5 is input

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=1) # Kaiming uniform for FPN convs
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, backbone_features):
        """
        Args:
            backbone_features (list[Tensor]): List of feature maps from the backbone.
                                             Ordered from shallower (e.g., C3) to deeper (e.g., C5).
                                             Example: [c3_feat, c4_feat, c5_feat]
        Returns:
            list[Tensor]: List of FPN feature maps (P-layers).
                          Ordered from shallower (e.g., P3) to deeper (e.g., P5, P6).
        """
        if len(backbone_features) != len(self.in_channels_list):
            raise ValueError(
                f"Number of backbone_features ({len(backbone_features)}) does not match "
                f"in_channels_list ({len(self.in_channels_list)})"
            )

        # Apply lateral convolutions to backbone features
        # Process from deepest to shallowest for top-down pathway
        # C5, C4, C3 (example names)
        # backbone_features[2] is C5, backbone_features[1] is C4, backbone_features[0] is C3

        # Lateral connections (from deepest C layer to shallowest)
        # m_i = lateral_conv_i(c_i)
        laterals = [
            lat_conv(c_feat) for lat_conv, c_feat in zip(self.lateral_convs, backbone_features)
        ]
        # laterals are now [m3, m4, m5] if backbone_features was [c3, c4, c5]

        # Top-down pathway
        # Start with the deepest lateral feature (e.g., m5)
        # P5 = fpn_conv5(m5)
        # P4 = fpn_conv4(m4 + Upsample(P5_intermediate_for_addition_or_m5))
        # P3 = fpn_conv3(m3 + Upsample(P4_intermediate_for_addition))

        num_backbone_levels = len(laterals)
        fpn_outputs = [None] * num_backbone_levels # To store P3, P4, P5

        # P5 (or deepest P layer)
        # The deepest lateral becomes the input to the deepest P-layer's 3x3 conv
        fpn_outputs[num_backbone_levels - 1] = self.fpn_convs[num_backbone_levels - 1](
            laterals[num_backbone_levels - 1]
        )

        # Iterate from the second deepest to the shallowest
        # P_i = fpn_conv_i(lateral_i + Upsample(P_{i+1}))
        for i in range(num_backbone_levels - 2, -1, -1):
            # Upsample the previous (deeper) FPN output to match current lateral's size
            # Note: F.interpolate needs size or scale_factor.
            # We assume a halving of sequence length at each backbone stage for scale_factor=2.
            # If pooling is MaxPool1d(3), sequence length is roughly divided by 3.
            # For FPN, it's common to use scale_factor=2 for upsampling,
            # relying on convs to adapt. If sizes mismatch significantly, this needs care.
            # Let's assume for now that upsampling by 2 is a reasonable approximation
            # or that the backbone stages are designed for this.
            # A more robust way is to use laterals[i].shape[-1] as target size for upsample.
            
            prev_fpn_output = fpn_outputs[i+1] # This is P_{i+1}
            target_size = laterals[i].shape[-1] # Sequence length of current lateral (m_i)
            
            upsampled_prev_fpn = F.interpolate(prev_fpn_output, size=target_size, mode='nearest') # or 'linear'
            
            # Add lateral connection
            summed_features = laterals[i] + upsampled_prev_fpn
            
            # Apply final 3x3 conv
            fpn_outputs[i] = self.fpn_convs[i](summed_features)

        # Add extra P6 layer if configured
        if self.extra_blocks:
            # P6 is made from the deepest FPN output (P5)
            p5_output = fpn_outputs[num_backbone_levels - 1]
            p6_output = self.p6_conv(p5_output) # Strided conv for P6
            # If using MaxPool1d for P6: p6_output = self.p6_pool(p5_output)
            fpn_outputs.append(p6_output)

        return fpn_outputs


# --- Detection Heads ---
class DetectionHead(nn.Module):
    def __init__(self,
                 num_classes: int,
                 fpn_out_channels: int,
                 num_anchors_per_level: int,
                 num_convs: int = 4, # Number of conv layers in each subnet
                 prior_probability: float = 0.01): # For initializing the bias of the last classification conv
        """
        Detection Head module.

        Args:
            num_classes (int): Number of target classes (excluding background).
            fpn_out_channels (int): Number of channels from FPN P-layers.
            num_anchors_per_level (int): Number of anchors associated with each spatial location
                                         on an FPN feature map.
            num_convs (int): Number of 3x3 Conv1d layers in each subnet (classification and regression).
            prior_probability (float): Used to initialize the bias of the final classification conv layer.
                                       Helps with training stability, especially with many negative samples.
        """
        super(DetectionHead, self).__init__()
        self.num_classes = num_classes
        self.fpn_out_channels = fpn_out_channels
        self.num_anchors_per_level = num_anchors_per_level
        self.num_convs = num_convs

        # Classification Subnet
        cls_subnet_layers = []
        for _ in range(num_convs):
            cls_subnet_layers.append(
                nn.Conv1d(fpn_out_channels, fpn_out_channels, kernel_size=3, stride=1, padding=1)
            )
            cls_subnet_layers.append(nn.ReLU())
        self.cls_subnet = nn.Sequential(*cls_subnet_layers)

        # Regression Subnet (for segment boundaries)
        reg_subnet_layers = []
        for _ in range(num_convs):
            reg_subnet_layers.append(
                nn.Conv1d(fpn_out_channels, fpn_out_channels, kernel_size=3, stride=1, padding=1)
            )
            reg_subnet_layers.append(nn.ReLU())
        self.reg_subnet = nn.Sequential(*reg_subnet_layers)

        # Final prediction layers
        # Classifier: predicts num_classes scores for each anchor
        self.cls_predictor = nn.Conv1d(
            fpn_out_channels, num_anchors_per_level * num_classes, kernel_size=3, stride=1, padding=1
        )
        # Regressor: predicts 2 values (e.g., start_offset, end_offset) for each anchor
        self.reg_predictor = nn.Conv1d(
            fpn_out_channels, num_anchors_per_level * 2, kernel_size=3, stride=1, padding=1
        ) # 2 values: delta_start, delta_end

        self._initialize_weights(prior_probability)

    def _initialize_weights(self, prior_probability: float):
        for m in self.cls_subnet.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.reg_subnet.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.cls_predictor.weight, std=0.01)
        # Initialize the bias of the classification predictor for stability (RetinaNet trick)
        bias_value = -math.log((1 - prior_probability) / prior_probability)
        nn.init.constant_(self.cls_predictor.bias, bias_value)

        nn.init.normal_(self.reg_predictor.weight, std=0.01)
        nn.init.constant_(self.reg_predictor.bias, 0)


    def forward(self, fpn_features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Apply detection heads to FPN features.

        Args:
            fpn_features (List[torch.Tensor]): List of feature maps from FPN (P3, P4, P5, ...).

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]:
                - all_cls_logits (List[Tensor]): List of classification logits for each FPN level.
                    Each tensor shape: [Batch, num_anchors * num_classes, SeqLength_level]
                - all_reg_preds (List[Tensor]): List of regression predictions for each FPN level.
                    Each tensor shape: [Batch, num_anchors * 2, SeqLength_level]
        """
        all_cls_logits = []
        all_reg_preds = []

        for feature_map in fpn_features:
            # Pass through classification subnet
            cls_feat = self.cls_subnet(feature_map)
            cls_logits = self.cls_predictor(cls_feat) # [B, num_anchors*num_classes, L]

            # Pass through regression subnet
            reg_feat = self.reg_subnet(feature_map)
            reg_pred = self.reg_predictor(reg_feat) # [B, num_anchors*2, L]

            all_cls_logits.append(cls_logits)
            all_reg_preds.append(reg_pred)

        return all_cls_logits, all_reg_preds


# --- Anchor Generator (Simplified for 1D) ---
class AnchorGenerator1D(nn.Module):
    def __init__(self,
                 base_anchor_lengths: List[List[int]], # e.g., [[30, 40, 50], [180, 200, 220], ...]
                 fpn_strides: List[int]): # Strides of FPN levels relative to input, e.g., [8, 16, 32]
        """
        Generates 1D anchors for each FPN level.

        Args:
            base_anchor_lengths (List[List[int]]):
                A list of lists. Each inner list contains the base anchor lengths
                for the corresponding FPN level.
                Example: If FPN has P3, P4, P5:
                         [[p3_len1, p3_len2], [p4_len1, p4_len2], [p5_len1, p5_len2]]
            fpn_strides (List[int]): The downsampling factor (stride) of each FPN level
                                     with respect to the original input sequence length.
                                     Needed to map anchor coordinates back to input space.
        """
        super(AnchorGenerator1D, self).__init__()
        if len(base_anchor_lengths) != len(fpn_strides):
            raise ValueError("Number of FPN levels for anchor lengths and strides must match.")

        self.base_anchor_lengths = base_anchor_lengths
        self.fpn_strides = fpn_strides
        self.num_anchors_per_level = [len(lengths) for lengths in base_anchor_lengths]

    def generate_anchors_for_level(self,
                                   feature_map_length: int,
                                   level_idx: int,
                                   device: torch.device) -> torch.Tensor:
        """
        Generates anchors for a single FPN level.
        Anchors are defined by their (start, end) coordinates in the original input sequence scale.
        """
        stride = self.fpn_strides[level_idx]
        lengths = self.base_anchor_lengths[level_idx]

        # Center positions of anchors on the feature map, scaled to input resolution
        # For each point on the feature map, its "center" in the input sequence is i * stride.
        # We can simplify and say the anchor starts around i * stride.
        # Let's define anchor centers more precisely.
        # A common practice is to center anchors at (i + 0.5) * stride for feature map cell i.
        
        centers_x = (torch.arange(0, feature_map_length, device=device) + 0.5) * stride
        # centers_x shape: [feature_map_length]

        anchors_level = []
        for length in lengths:
            half_len = length / 2.0
            starts = centers_x - half_len
            ends = centers_x + half_len
            # Stack to get [feature_map_length, 2] where 2 is (start, end)
            anchors_for_this_length = torch.stack([starts, ends], dim=1)
            anchors_level.append(anchors_for_this_length)

        # Concatenate anchors for all base lengths at this level
        # Result shape: [feature_map_length * num_base_anchors_this_level, 2]
        anchors_level_tensor = torch.cat(anchors_level, dim=0)
        return anchors_level_tensor.round() # Round to nearest integer for segment boundaries

    def forward(self, fpn_feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Generate anchors for all FPN levels.

        Args:
            fpn_feature_maps (List[torch.Tensor]): List of feature maps from FPN (P3, P4, P5, ...).
                                                   Used to get sequence lengths.

        Returns:
            List[torch.Tensor]: A list of anchor tensors. Each tensor corresponds to an FPN level
                                and has shape [NumAnchorsAtLevel, 2] (start, end).
                                NumAnchorsAtLevel = feature_map_length * num_base_anchors_this_level
        """
        anchors_all_levels = []
        for level_idx, feature_map in enumerate(fpn_feature_maps):
            feature_map_length = feature_map.shape[-1]
            anchors_level = self.generate_anchors_for_level(
                feature_map_length, level_idx, feature_map.device
            )
            anchors_all_levels.append(anchors_level)
        return anchors_all_levels


# --- Complete Detection Model ---
class ObjectDetector1D(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 fpn: nn.Module,
                 anchor_generator: AnchorGenerator1D,
                 detection_head: DetectionHead,
                 num_classes: int,
                 # Loss parameters
                 matcher_high_threshold: float = 0.5,
                 matcher_low_threshold: float = 0.4,
                 focal_loss_alpha: float = 0.25,
                 focal_loss_gamma: float = 2.0,
                 regression_loss_weight: float = 1.0):
        super(ObjectDetector1D, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.anchor_generator = anchor_generator
        self.detection_head = detection_head
        self.num_classes = num_classes

        self.loss_computer = DetectionLoss(
            num_classes=num_classes,
            matcher_high_threshold=matcher_high_threshold,
            matcher_low_threshold=matcher_low_threshold,
            focal_loss_alpha=focal_loss_alpha,
            focal_loss_gamma=focal_loss_gamma,
            regression_loss_weight=regression_loss_weight
        )
        # Pass num_anchors_per_level to loss computer
        self.loss_computer.set_num_anchors_per_level(self.anchor_generator.num_anchors_per_level)


    def forward(self, sequences: torch.Tensor, targets: Optional[List[dict]] = None):
        # 1. Get features from backbone
        backbone_features = self.backbone(sequences) # List of Tensors

        # 2. Get FPN features
        fpn_features = self.fpn(backbone_features) # List of Tensors [P3, P4, P5, P6]

        # 3. Get predictions from detection head
        cls_logits_list, reg_preds_list = self.detection_head(fpn_features)
        # cls_logits_list: List of [B, A_lvl*C, L_lvl]
        # reg_preds_list: List of [B, A_lvl*2, L_lvl]

        # 4. Generate anchors
        # anchors_list: List of [NumTotalAnchorsAtLevel, 2] (start, end)
        # These are generated based on feature map shapes from fpn_features,
        # so they are consistent for the current batch.
        anchors_list = self.anchor_generator(fpn_features)

        if self.training and targets is not None:
            # Compute losses
            classification_loss, regression_loss = self.loss_computer(
                cls_logits_list,
                reg_preds_list,
                anchors_list,
                targets
            )
            total_loss = classification_loss + regression_loss
            return {"classification_loss": classification_loss,
                    "regression_loss": regression_loss,
                    "total_loss": total_loss}
        else:
            # Inference mode: Decode predictions and apply NMS
            # This part needs to be implemented for actual detection output.
            # For now, returning raw outputs as per your original design.
            # You would typically decode reg_preds_list using anchors_list,
            # apply sigmoid to cls_logits_list, then NMS.
            return cls_logits_list, reg_preds_list, anchors_list, fpn_features
        

if __name__=="__main__":
##############DetectionHead test###################################################################################

    NUM_CLASSES = 100 # Your number of sequence classes
    FPN_COMMON_CHANNELS = 128 # Channels for FPN P-layers and input to heads

    # 1. Backbone Configuration
    cfg_1d_backbone = {'N': [128, 128, 'M', 256, 256, 'M', 512]} # C3=128, C4=256, C5=512 channels
    backbone = RF_Backbone(cfg_1d=cfg_1d_backbone['N'])

    # Determine backbone output channels (must match FPN in_channels_list)
    # This is a bit manual; ideally, backbone would expose this.
    # Based on cfg: C3->128, C4->256, C5->512
    backbone_output_channel_list = [128, 256, 512] # Corresponds to outputs of RF_Backbone

    # 2. FPN
    # Assuming RF_Backbone outputs 3 feature maps (like C3, C4, C5)
    # If RF_Backbone.feature_stages_1d has 3 stages, then len(backbone_output_channel_list) is 3
    fpn = FeaturePyramidNetwork1D(
        in_channels_list=backbone_output_channel_list,
        out_channels=FPN_COMMON_CHANNELS,
        extra_blocks=True # To get P6 from P5
    )
    # FPN will output P3, P4, P5, P6. All with FPN_COMMON_CHANNELS.

    # 3. Anchor Generator
    # Strides: Need to calculate these based on your backbone's pooling.
    # initial_2d_processor: MaxPool2d((1,3)), MaxPool2d((2,2)) -> total stride_2d_seq = 3*2 = 6
    # feature_stages_1d: MaxPool1d(3) after C3, MaxPool1d(3) after C4
    # Stride for P3 (output of first 1D stage, before its 'M'): initial_2d_seq_stride * (1 if no pool before it)
    # Stride for P4 (output of second 1D stage): initial_2d_seq_stride * 3
    # Stride for P5 (output of third 1D stage): initial_2d_seq_stride * 3 * 3
    # Stride for P6 (from P5 with stride 2 conv): initial_2d_seq_stride * 3 * 3 * 2
    # Let's assume effective strides for P3, P4, P5, P6 for simplicity in example.
    # These strides are CRITICAL and must match how feature maps shrink.
    # For RF_Backbone:
    # - initial_2d_processor has MaxPool2d((1,3)) then MaxPool2d((2,2)). Seq stride = 3 * 2 = 6.
    # - feature_stages_1d:
    #   - Stage 1 (for P3): No MaxPool1d *inside* the first block of layers in cfg. Output is C3. Stride = 6.
    #   - Stage 2 (for P4): MaxPool1d(3) after C3. Output is C4. Stride = 6 * 3 = 18.
    #   - Stage 3 (for P5): MaxPool1d(3) after C4. Output is C5. Stride = 18 * 3 = 54.
    # - FPN P6 is from P5 with stride 2: Stride = 54 * 2 = 108.
    fpn_strides_calculated = [6, 18, 54, 108] # For P3, P4, P5, P6 respectively

    # Anchor lengths for each FPN level (P3, P4, P5, P6)
    # You can have multiple anchor lengths per level.
    # Example: P3 for small, P4 for medium, P5 for large, P6 for very large.
    # Your initial: P3:30, P4:200, P5:2000. Let's add P6 and multiple anchors.
    anchor_lengths_per_level = [
        [30, 40, 50],       # P3: for small segments (lengths around 20-40)
        [200, 300, 450],    # P4: for medium segments (lengths around 150-250)
        [1500, 2000, 2500], # P5: for large segments (lengths around 1500-2500)
        [4000, 5000],       # P6: for very large segments (lengths around 4000-5000)
    ]
    # Number of anchors per location on each FPN level
    num_anchors_per_fpn_level = [len(lengths) for lengths in anchor_lengths_per_level]
    # Ensure this matches the number of FPN output features (P3, P4, P5, P6 -> 4 levels)
    if len(anchor_lengths_per_level) != len(fpn_strides_calculated):
         raise ValueError("Mismatch between anchor_lengths_per_level and fpn_strides_calculated definitions")


    anchor_generator = AnchorGenerator1D(
        base_anchor_lengths=anchor_lengths_per_level,
        fpn_strides=fpn_strides_calculated
    )

    # 4. Detection Head
    # num_anchors_per_level for the head should be the number of different anchor *types*
    # (e.g., different lengths) you use at each location.
    # If anchor_lengths_per_level = [[l1,l2],[l3,l4]...], then num_anchors_per_level for the head
    # should be consistent. The AnchorGenerator1D handles generating all of them.
    # The DetectionHead predicts for *each* of these num_anchors_per_level.
    # So, if P3 has 3 anchor lengths [20,30,40], num_anchors_per_level for the head is 3.
    # We assume all FPN levels use the same number of base anchor types for simplicity in head design.
    # If not, the head or its output reshaping needs to be more complex.
    # Let's assume the head is designed for a fixed number of anchor variants per location,
    # and this number is consistent across FPN levels.
    # The current DetectionHead assumes num_anchors_per_level is a single int.
    # This means each FPN level should ideally have the same number of base anchors.
    # Let's adjust anchor_lengths_per_level to have same number of anchors, e.g., 3 per level.
    anchor_lengths_per_level_uniform = [
        [30, 40, 50],       # P3
        [200, 300, 400],    # P4
        [1500, 2000, 2500], # P5
        [4000, 5000, 6000]  # P6 (added one to make it 3)
    ]
    num_base_anchors = len(anchor_lengths_per_level_uniform[0]) # Should be 3

    anchor_generator_uniform = AnchorGenerator1D( # Re-init with uniform number of anchors
        base_anchor_lengths=anchor_lengths_per_level_uniform,
        fpn_strides=fpn_strides_calculated
    )

    detection_head = DetectionHead(
        num_classes=NUM_CLASSES,
        fpn_out_channels=FPN_COMMON_CHANNELS,
        num_anchors_per_level=num_base_anchors # e.g., 3 anchor lengths per location
    )

    # 5. Assemble the full model
    model = ObjectDetector1D(
        backbone=backbone,
        fpn=fpn,
        anchor_generator=anchor_generator_uniform, # Use the one with uniform anchors
        detection_head=detection_head,
        num_classes=NUM_CLASSES
    )
    print(model)

    # --- Test Forward Pass ---
    batch_size = 2
    sequence_length = 8000 # Example sequence length
    dummy_sequences = torch.randn(batch_size, 2, sequence_length)

    model.eval() # Set to evaluation mode for anchor generation and inference logic
    with torch.no_grad():
        cls_logits_list, reg_preds_list, anchors_list, _ = model(dummy_sequences)

    print("\n--- Model Output ---")
    print(f"Number of FPN levels processed: {len(cls_logits_list)}")

    for i in range(len(cls_logits_list)):
        print(f"\nFPN Level {i+1} (e.g., P{i+3}):")
        print(f"  Class logits shape: {cls_logits_list[i].shape}")
        # Expected: [Batch, num_base_anchors * NUM_CLASSES, SeqLength_level_i]
        print(f"  Regression preds shape: {reg_preds_list[i].shape}")
        # Expected: [Batch, num_base_anchors * 2, SeqLength_level_i]
        print(f"  Anchors generated for this level shape: {anchors_list[i].shape}")
        # Expected: [SeqLength_level_i * num_base_anchors, 2]

    # Example of how to reshape predictions for easier processing:
    # For the first FPN level's classification logits:
    # cls_logits_p3 = cls_logits_list[0] # [B, num_base_anchors * NUM_CLASSES, L_p3]
    # B, _, L_p3 = cls_logits_p3.shape
    # cls_logits_p3_reshaped = cls_logits_p3.view(B, num_base_anchors, NUM_CLASSES, L_p3)
    # cls_logits_p3_permuted = cls_logits_p3_reshaped.permute(0, 3, 1, 2).contiguous()
    # cls_logits_p3_final = cls_logits_p3_permuted.view(B, L_p3 * num_base_anchors, NUM_CLASSES)
    # print(f"\nExample reshaped cls_logits for P3 (all anchors): {cls_logits_p3_final.shape}")
    # This final shape [B, TotalAnchorsAtLevel, NUM_CLASSES] is often useful for loss computation/NMS.
    # Similar reshaping can be done for regression predictions.