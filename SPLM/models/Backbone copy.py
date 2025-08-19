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
'''
class ObjectDetector1D(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 fpn: nn.Module,
                 anchor_generator: AnchorGenerator1D,
                 detection_head: DetectionHead,
                 num_classes: int):
        super(ObjectDetector1D, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.anchor_generator = anchor_generator
        self.detection_head = detection_head
        self.num_classes = num_classes # For reference, already in detection_head

    def forward(self, sequences: torch.Tensor, targets: List[dict] = None):
        """
        Args:
            sequences (Tensor): Input batch of sequences, shape [Batch, 2, SequenceLength].
            targets (List[dict], optional): Ground truth targets for training.
                                            Each dict could contain 'segments' and 'labels'.
                                            If None, operates in inference mode.
        Returns:
            During training (if targets are provided and loss is computed here):
                dict: A dictionary of losses.
            During inference:
                List[Tuple[Tensor, Tensor, Tensor]]: For each image in the batch, a tuple of
                    (detected_segments, detected_scores, detected_labels).
                    Or, more directly, the raw outputs from detection_head for post-processing.
        """
        # 1. Get features from backbone
        backbone_features = self.backbone(sequences)

        # 2. Get FPN features
        fpn_features = self.fpn(backbone_features)
        # fpn_features is a list like [P3_feat, P4_feat, P5_feat, P6_feat]

        # 3. Get predictions from detection head
        # cls_logits_list: List of [B, num_anchors_per_loc*num_classes, L_level]
        # reg_preds_list: List of [B, num_anchors_per_loc*2, L_level]
        cls_logits_list, reg_preds_list = self.detection_head(fpn_features)

        # 4. Generate anchors (anchors are generated on the fly based on feature map sizes)
        # This is usually done once or can be pre-computed if input sizes are fixed.
        # For dynamic input sizes, it's done per forward pass.
        # anchors_list: List of [NumTotalAnchorsAtLevel, 2] (start, end)
        anchors_list = self.anchor_generator(fpn_features)

        # At this point, for training, you would:
        # - Match anchors to ground truth targets.
        # - Compute classification loss (e.g., Focal Loss) and regression loss (e.g., Smooth L1).
        # - Return the losses.

        # For inference, you would:
        # - Apply sigmoid to cls_logits_list to get probabilities.
        # - Decode reg_preds_list with anchors_list to get actual segment coordinates.
        # - Perform NMS (Non-Maximum Suppression).
        # - Return detected segments, scores, and labels.

        # For now, let's return the raw predictions and anchors for external processing.
        return cls_logits_list, reg_preds_list, anchors_list, fpn_features
'''

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
        
# --- 辅助函数 ---
def calculate_iou_1d(segments1: torch.Tensor, segments2: torch.Tensor) -> torch.Tensor:
    """
    Calculate 1D Intersection over Union (IoU) between two sets of segments.
    Args:
        segments1 (Tensor): Shape [N, 2], where N is the number of segments. Each row is [start, end].
        segments2 (Tensor): Shape [M, 2], where M is the number of segments. Each row is [start, end].
    Returns:
        Tensor: Shape [N, M], the IoU matrix.
    """
    # Expand dimensions to allow broadcasting
    # segments1: [N, 1, 2]
    # segments2: [1, M, 2]
    segments1 = segments1.unsqueeze(1)
    segments2 = segments2.unsqueeze(0)

    # Calculate intersection
    # max_start: [N, M]
    # min_end:   [N, M]
    max_start = torch.max(segments1[..., 0], segments2[..., 0])
    min_end = torch.min(segments1[..., 1], segments2[..., 1])
    
    intersection_lengths = torch.clamp(min_end - max_start, min=0) # [N, M]

    # Calculate union
    lengths1 = segments1[..., 1] - segments1[..., 0] # [N, 1]
    lengths2 = segments2[..., 1] - segments2[..., 0] # [1, M]
    union_lengths = lengths1 + lengths2 - intersection_lengths # [N, M]

    # IoU
    iou = intersection_lengths / torch.clamp(union_lengths, min=1e-6) # Avoid division by zero
    return iou


class Matcher:
    """
    Assigns ground truth segments to anchors based on IoU.
    """
    def __init__(self, high_threshold: float, low_threshold: float, allow_low_quality_matches: bool = True):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, anchors: torch.Tensor, gt_segments: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            anchors (Tensor): Generated anchors, shape [num_anchors, 2].
            gt_segments (Tensor): Ground truth segments, shape [num_gt, 2].
        Returns:
            Tuple[Tensor, Tensor]:
                - labels (Tensor[int64]): For each anchor, -1 for ignore, 0 for background (negative),
                                          1 for foreground (positive). Shape [num_anchors].
                - matched_gt_idxs (Tensor[int64]): For each anchor, the index of the matched GT segment,
                                                 or -1 if not a positive match. Shape [num_anchors].
        """
        num_anchors = anchors.shape[0]
        
        if gt_segments.numel() == 0: # No ground truth segments
            labels = torch.zeros(num_anchors, dtype=torch.int64, device=anchors.device)
            matched_gt_idxs = torch.full((num_anchors,), -1, dtype=torch.int64, device=anchors.device)
            return labels, matched_gt_idxs

        iou_matrix = calculate_iou_1d(anchors, gt_segments) # [num_anchors, num_gt]

        # Find the best GT for each anchor
        # best_gt_iou_for_anchor: [num_anchors], best_gt_idx_for_anchor: [num_anchors]
        best_gt_iou_for_anchor, best_gt_idx_for_anchor = iou_matrix.max(dim=1)

        # Initialize labels: 0 for background, -1 for ignore
        labels = torch.full((num_anchors,), -1, dtype=torch.int64, device=anchors.device)
        
        # Assign negatives
        labels[best_gt_iou_for_anchor < self.low_threshold] = 0
        
        # Assign positives based on high_threshold
        positive_mask = best_gt_iou_for_anchor >= self.high_threshold
        labels[positive_mask] = 1
        
        # Assign positives based on allow_low_quality_matches (force match for each GT)
        if self.allow_low_quality_matches:
            # Find the best anchor for each GT
            # best_anchor_iou_for_gt: [num_gt]
            best_anchor_iou_for_gt, _ = iou_matrix.max(dim=0) 
            
            for gt_idx in range(gt_segments.shape[0]):
                # Anchors that have max IoU with this GT
                match_with_this_gt_mask = (iou_matrix[:, gt_idx] == best_anchor_iou_for_gt[gt_idx]) & \
                                          (iou_matrix[:, gt_idx] > 0) # Ensure some overlap
                labels[match_with_this_gt_mask] = 1
        
        # Store matched GT indices for positive anchors
        matched_gt_idxs = torch.full((num_anchors,), -1, dtype=torch.int64, device=anchors.device)
        # For anchors labeled as positive, store the index of the GT they matched best with
        # Note: if allow_low_quality_matches changed an anchor to positive, its best_gt_idx_for_anchor
        # might not correspond to the GT that *forced* the match. A more robust way is to re-check.
        # For simplicity here, we use best_gt_idx_for_anchor.
        # A more correct assignment for forced matches:
        # if self.allow_low_quality_matches:
        #     for gt_idx in range(gt_segments.shape[0]):
        #         # Find anchors that have max IoU with this GT and are positive
        #         force_match_mask = (iou_matrix[:, gt_idx] == best_anchor_iou_for_gt[gt_idx]) & \
        #                            (iou_matrix[:, gt_idx] > 0) & (labels == 1)
        #         matched_gt_idxs[force_match_mask] = gt_idx
        # else: # Original assignment if not using forced matches or for those already above high_threshold
        #     positive_mask_final = labels == 1
        #     matched_gt_idxs[positive_mask_final] = best_gt_idx_for_anchor[positive_mask_final]
        
        # Simpler assignment: if an anchor is positive, it's matched to the GT it has highest IoU with.
        # This is generally sufficient.
        positive_mask_final = labels == 1
        if positive_mask_final.any():
             matched_gt_idxs[positive_mask_final] = best_gt_idx_for_anchor[positive_mask_final]

        return labels, matched_gt_idxs


def encode_regression_targets(anchors: torch.Tensor, gt_segments: torch.Tensor) -> torch.Tensor:
    """
    Encode ground truth segments into regression targets (deltas) relative to anchors.
    Args:
        anchors (Tensor): Positive anchors, shape [num_positive_anchors, 2] (start, end).
        gt_segments (Tensor): Corresponding matched ground truth segments, shape [num_positive_anchors, 2].
    Returns:
        Tensor: Regression targets (delta_start, delta_end), shape [num_positive_anchors, 2].
    """
    # Simple difference encoding
    # You might want to normalize by anchor length, e.g.,
    # anchor_lengths = (anchors[:, 1] - anchors[:, 0]).unsqueeze(1) + 1e-6
    # delta_start = (gt_segments[:, 0] - anchors[:, 0]) / anchor_lengths
    # delta_end = (gt_segments[:, 1] - anchors[:, 1]) / anchor_lengths
    # return torch.stack([delta_start, delta_end], dim=1)
    
    delta_start = gt_segments[:, 0] - anchors[:, 0]
    delta_end = gt_segments[:, 1] - anchors[:, 1]
    return torch.stack([delta_start, delta_end], dim=1).squeeze(-1) # Ensure [N, 2]

# --- Focal Loss (from torchvision, adapted slightly if needed, or use a library version) ---
# For simplicity, we'll use BCEWithLogitsLoss for classification first,
# then show how Focal Loss could be integrated.
# A common Focal Loss implementation:
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (Tensor): Logits, shape [N, C] or [N].
            targets (Tensor): Ground truth labels, shape [N, C] (one-hot) or [N] (class indices).
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when BCE_loss is large
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# --- Loss Computation Class ---
class DetectionLoss(nn.Module):
    def __init__(self, num_classes: int, 
                 matcher_high_threshold: float = 0.5,
                 matcher_low_threshold: float = 0.4,
                 focal_loss_alpha: float = 0.25,
                 focal_loss_gamma: float = 2.0,
                 regression_loss_weight: float = 1.0):
        super(DetectionLoss, self).__init__()
        self.num_classes = num_classes
        self.matcher = Matcher(matcher_high_threshold, matcher_low_threshold)
        self.classification_loss_fn = FocalLoss(alpha=focal_loss_alpha, gamma=focal_loss_gamma)
        # self.classification_loss_fn = nn.BCEWithLogitsLoss() # Simpler alternative
        self.regression_loss_fn = nn.SmoothL1Loss(beta=1.0/9.0) # beta is common for SmoothL1 in detection
        self.regression_loss_weight = regression_loss_weight

    def forward(self,
                cls_logits_list: List[torch.Tensor],  # List of [B, A*C, L_level]
                reg_preds_list: List[torch.Tensor],   # List of [B, A*2, L_level]
                anchors_list: List[torch.Tensor],     # List of [N_anchors_level, 2] (NOT per batch item yet)
                targets: List[Dict[str, torch.Tensor]] # List of dicts, one per batch item
               ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size = cls_logits_list[0].shape[0]
        
        # Concatenate predictions and anchors from all FPN levels
        # Permute and reshape logits and preds to be [Batch, TotalAnchors, NumFeatures]
        # cls_logits: [B, sum_A_L, C]
        # reg_preds: [B, sum_A_L, 2]
        
        processed_cls_logits = []
        processed_reg_preds = []
        
        # Example: cls_logits_list[0] is [B, num_anchors_per_loc * num_classes, L0]
        # num_anchors_per_loc is implicitly defined by your AnchorGenerator1D's base_anchor_lengths
        # For now, let's assume num_anchors_per_loc is consistent or can be derived.
        # The AnchorGenerator1D.num_anchors_per_level gives num_base_lengths for that level.
        # So, A in A*C is num_base_lengths.
        
        for i_level in range(len(cls_logits_list)):
            cls_l = cls_logits_list[i_level] # [B, num_base_lengths_lvl * C, L_lvl]
            reg_l = reg_preds_list[i_level]  # [B, num_base_lengths_lvl * 2, L_lvl]
            num_base_lengths_lvl = self.num_anchors_per_level[i_level] # Needs to be passed or inferred

            # Reshape for easier processing:
            # cls_l: [B, L_lvl, num_base_lengths_lvl, C] -> [B, L_lvl * num_base_lengths_lvl, C]
            cls_l = cls_l.permute(0, 2, 1) # [B, L_lvl, num_base_lengths_lvl * C]
            cls_l = cls_l.reshape(batch_size, -1, self.num_classes) # [B, TotalAnchors_lvl, C]
            processed_cls_logits.append(cls_l)

            # reg_l: [B, L_lvl, num_base_lengths_lvl, 2] -> [B, L_lvl * num_base_lengths_lvl, 2]
            reg_l = reg_l.permute(0, 2, 1) # [B, L_lvl, num_base_lengths_lvl * 2]
            reg_l = reg_l.reshape(batch_size, -1, 2) # [B, TotalAnchors_lvl, 2]
            processed_reg_preds.append(reg_l)
            
        all_cls_logits = torch.cat(processed_cls_logits, dim=1) # [B, TotalAnchorsAcrossLevels, C]
        all_reg_preds = torch.cat(processed_reg_preds, dim=1)   # [B, TotalAnchorsAcrossLevels, 2]
        
        # Anchors are currently List[[N_anchors_level, 2]]. Concatenate them.
        # These anchors are the same for all items in the batch.
        concatenated_anchors = torch.cat(anchors_list, dim=0) # [TotalAnchorsAcrossLevels, 2]

        # Initialize lists to store per-batch-item targets
        batched_cls_targets = []
        batched_reg_targets = []
        positive_anchor_indices_for_reg = []
        
        num_total_anchors = concatenated_anchors.shape[0]

        for i in range(batch_size):
            gt_segments_i = targets[i]['segments'] # [num_gt_i, 2]
            gt_labels_i = targets[i]['labels']     # [num_gt_i]

            # Match anchors with GT for this specific image
            # anchor_match_labels: [TotalAnchors], -1 ignore, 0 neg, 1 pos
            # matched_gt_indices: [TotalAnchors], index of GT for positive anchors
            anchor_match_labels, matched_gt_indices = self.matcher(concatenated_anchors, gt_segments_i)

            # Prepare classification targets
            # Target for Focal Loss: one-hot encoded, or 0/1 for BCEWithLogitsLoss per class
            # cls_target_i: [TotalAnchors, NumClasses]
            cls_target_i = torch.zeros_like(all_cls_logits[i]) # [TotalAnchors, NumClasses]
            
            positive_mask = (anchor_match_labels == 1)
            # For positive anchors, set the target class to 1
            if positive_mask.any():
                matched_gt_labels_for_pos_anchors = gt_labels_i[matched_gt_indices[positive_mask]]
                cls_target_i[positive_mask, matched_gt_labels_for_pos_anchors] = 1.0
            
            # Negative anchors (anchor_match_labels == 0) already have all-zero targets in cls_target_i
            # Ignored anchors (anchor_match_labels == -1) will be masked out later.
            
            batched_cls_targets.append(cls_target_i)

            # Prepare regression targets (only for positive anchors)
            if positive_mask.any():
                positive_anchors = concatenated_anchors[positive_mask]
                matched_gt_segments_for_pos_anchors = gt_segments_i[matched_gt_indices[positive_mask]]
                
                reg_target_i = encode_regression_targets(positive_anchors, matched_gt_segments_for_pos_anchors)
                batched_reg_targets.append(reg_target_i)
                positive_anchor_indices_for_reg.append(positive_mask.nonzero(as_tuple=False).squeeze(1) + i * num_total_anchors) # Global indices for batch
            
        # --- Classification Loss ---
        # Stack batched targets: [B, TotalAnchors, C]
        final_cls_targets = torch.stack(batched_cls_targets, dim=0)
        
        # Mask for anchors that are not "ignore"
        # anchor_match_labels is per item, need to reconstruct for batch or process per item
        # For simplicity, let's assume FocalLoss handles ignore internally or we filter inputs.
        # Here, we compute loss over all anchors (pos and neg), assuming FocalLoss handles imbalance.
        # If using BCE, ensure only pos/neg are included.
        # Valid mask for classification (not ignored anchors)
        # This needs to be constructed carefully if anchor_match_labels were per item.
        # Let's re-evaluate: all_cls_logits is [B, TotalAnchors, C]
        # final_cls_targets is [B, TotalAnchors, C]
        # We need a mask for valid anchors for classification: [B, TotalAnchors]
        # This requires re-running matcher or storing anchor_match_labels per batch item.
        
        # Simplified: Assume all_cls_logits and final_cls_targets are for *all* anchors.
        # FocalLoss expects logits and targets.
        # We need to select only non-ignored anchors for loss calculation.
        # Let's re-do the loop slightly to collect valid logits/targets.

        valid_cls_logits = []
        valid_cls_targets = []
        valid_reg_preds = []
        valid_reg_targets = []
        
        total_positive_anchors = 0

        for i in range(batch_size):
            gt_segments_i = targets[i]['segments']
            gt_labels_i = targets[i]['labels']
            anchor_match_labels, matched_gt_indices = self.matcher(concatenated_anchors, gt_segments_i) # [TotalAnchors]

            # Classification
            cls_valid_mask = (anchor_match_labels >= 0) # Positive (1) or Negative (0)
            if cls_valid_mask.any():
                valid_cls_logits.append(all_cls_logits[i][cls_valid_mask]) # [NumValidAnchors_i, C]
                
                cls_target_i_item = torch.zeros((cls_valid_mask.sum(), self.num_classes), device=all_cls_logits.device)
                
                positive_mask_item = (anchor_match_labels[cls_valid_mask] == 1) # Relative to cls_valid_mask
                if positive_mask_item.any():
                    # Get GT labels for these positive anchors
                    # Indices into original matched_gt_indices where cls_valid_mask is true AND anchor_match_label is 1
                    original_indices_of_positives = cls_valid_mask.nonzero(as_tuple=True)[0][positive_mask_item]
                    gt_labels_for_positives = gt_labels_i[matched_gt_indices[original_indices_of_positives]]
                    
                    # cls_target_i_item has shape [NumValidAnchors_i, C]
                    # positive_mask_item has shape [NumValidAnchors_i]
                    # gt_labels_for_positives has shape [NumPositiveAnchors_i]
                    cls_target_i_item[positive_mask_item, gt_labels_for_positives] = 1.0

                valid_cls_targets.append(cls_target_i_item)

            # Regression
            reg_valid_mask = (anchor_match_labels == 1) # Only Positive
            if reg_valid_mask.any():
                total_positive_anchors += reg_valid_mask.sum().item()
                valid_reg_preds.append(all_reg_preds[i][reg_valid_mask]) # [NumPositiveAnchors_i, 2]
                
                positive_anchors_item = concatenated_anchors[reg_valid_mask]
                matched_gt_segments_item = gt_segments_i[matched_gt_indices[reg_valid_mask]]
                reg_target_i_item = encode_regression_targets(positive_anchors_item, matched_gt_segments_item)
                valid_reg_targets.append(reg_target_i_item)

        if not valid_cls_logits: # No valid anchors found in the batch
            # Return zero loss or handle as an error/special case
            return torch.tensor(0.0, device=all_cls_logits.device, requires_grad=True), \
                   torch.tensor(0.0, device=all_reg_preds.device, requires_grad=True)

        final_cls_logits = torch.cat(valid_cls_logits, dim=0) # [TotalValidAnchorsInBatch, C]
        final_cls_targets = torch.cat(valid_cls_targets, dim=0) # [TotalValidAnchorsInBatch, C]
        
        classification_loss = self.classification_loss_fn(final_cls_logits, final_cls_targets)
        
        # --- Regression Loss ---
        if not valid_reg_preds: # No positive anchors in the batch
            regression_loss = torch.tensor(0.0, device=all_reg_preds.device)
        else:
            final_reg_preds = torch.cat(valid_reg_preds, dim=0) # [TotalPositiveAnchorsInBatch, 2]
            final_reg_targets = torch.cat(valid_reg_targets, dim=0) # [TotalPositiveAnchorsInBatch, 2]
            regression_loss = self.regression_loss_fn(final_reg_preds, final_reg_targets)
            # Normalize regression loss by number of positive anchors (common practice)
            if total_positive_anchors > 0:
                 regression_loss = regression_loss / total_positive_anchors 
            else: # Should not happen if valid_reg_preds is not empty
                 regression_loss = torch.tensor(0.0, device=all_reg_preds.device)


        return classification_loss, self.regression_loss_weight * regression_loss

    def set_num_anchors_per_level(self, num_anchors_per_level: List[int]):
        """Helper to pass num_anchors_per_level from AnchorGenerator."""
        self.num_anchors_per_level = num_anchors_per_level

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