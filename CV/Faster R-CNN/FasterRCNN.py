from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import feature_extraction

class DetectorBackboneWithFPN(nn.Module):
    """
    Detection backbone network: A tiny RegNet model coupled with a Feature
    Pyramid Network (FPN). This model takes in batches of input images with
    shape `(B, 3, H, W)` and gives features from three different FPN levels
    with shapes and total strides upto that level:

        - level p3: (out_channels, H /  8, W /  8)      stride =  8
        - level p4: (out_channels, H / 16, W / 16)      stride = 16
        - level p5: (out_channels, H / 32, W / 32)      stride = 32

    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

        # Initialize with ImageNet pre-trained weights.
        _cnn = models.regnet_x_400mf(pretrained=True)

        self.backbone = feature_extraction.create_feature_extractor(
            _cnn,
            return_nodes={
                "trunk_output.block2": "c3",
                "trunk_output.block3": "c4",
                "trunk_output.block4": "c5",
            },
        )

        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))
        dummy_out_shapes = [(key, value.shape) for key, value in dummy_out.items()]

        print("For dummy input images with shape: (2, 3, 224, 224)")
        for level_name, feature_shape in dummy_out_shapes:
            print(f"Shape of {level_name} features: {feature_shape}")

        ######################################################################
        # TODO: Initialize additional Conv layers for FPN.                   #
        #                                                                    #
        # Create THREE "lateral" 1x1 conv layers to transform (c3, c4, c5)   #
        # such that they all end up with the same `out_channels`.            #
        # Then create THREE "output" 3x3 conv layers to transform the merged #
        # FPN features to output (p3, p4, p5) features.                      #
        # All conv layers must have stride=1 and padding such that features  #
        # do not get downsampled due to 3x3 convs.                           #
        #                                                                    #
        # HINT: You have to use `dummy_out_shapes` defined above to decide   #
        # the input/output channels of these layers.                         #
        ######################################################################
        # This behaves like a Python dict, but makes PyTorch understand that
        # there are trainable weights inside it.
        # Add THREE lateral 1x1 conv and THREE output 3x3 conv layers.
        self.fpn_params = nn.ModuleDict()

        # Replace "pass" statement with your code
        self.fpn_params.update({
            'c3_lateral': nn.Conv2d(in_channels=dummy_out['c3'].shape[1], out_channels=self.out_channels,
                                    kernel_size=(1, 1)),
            'c3_output': nn.Conv2d(self.out_channels, self.out_channels, (3, 3), padding=1),

            'c4_lateral': nn.Conv2d(dummy_out['c4'].shape[1], self.out_channels, (1, 1)),
            'c4_output': nn.Conv2d(self.out_channels, self.out_channels, (3, 3), padding=1),

            'c5_lateral': nn.Conv2d(dummy_out['c5'].shape[1], self.out_channels, (1, 1)),
            'c5_output': nn.Conv2d(self.out_channels, self.out_channels, (3, 3), padding=1),
        })
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    @property
    def fpn_strides(self):
        """
        Total stride up to the FPN level. For a fixed ConvNet, these values
        are invariant to input image size. You may access these values freely
        to implement your logic in FCOS / Faster R-CNN.
        """
        return {"p3": 8, "p4": 16, "p5": 32}

    def forward(self, images: torch.Tensor):

        # Multi-scale features, dictionary with keys: {"c3", "c4", "c5"}.
        backbone_feats = self.backbone(images)

        fpn_feats = {"p3": None, "p4": None, "p5": None}
        ######################################################################
        # TODO: Fill output FPN features (p3, p4, p5) using RegNet features  #
        # (c3, c4, c5) and FPN conv layers created above.                    #
        # HINT: Use `F.interpolate` to upsample FPN features.                #
        ######################################################################

        # Replace "pass" statement with your code
        c3, c4, c5 = backbone_feats['c3'], backbone_feats['c4'], backbone_feats['c5']

        c5_lat = self.fpn_params["c5_lateral"](c5)  # Unify Channels
        p5 = self.fpn_params["c5_output"](c5_lat)
        p5_upsample = F.interpolate(input=p5, scale_factor=2.0,
                                    mode="bilinear")  # Call the Upsample API without implement it.
        c4_lat = self.fpn_params["c4_lateral"](c4)
        p4 = self.fpn_params["c4_output"](c4_lat + p5_upsample)  # Fuse
        p4_upsample = F.interpolate(input=p4, scale_factor=2.0, mode="bilinear")
        c3_lat = self.fpn_params["c3_lateral"](c3)
        p3 = self.fpn_params["c3_output"](c3_lat + p4_upsample)

        fpn_feats['p3'], fpn_feats['p4'], fpn_feats['p5'] = p3, p4, p5
        # Get more segment inf between dif scale picture.
        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

        return fpn_feats







def train_detector(
    detector,
    train_loader,
    learning_rate: float = 5e-3,
    weight_decay: float = 1e-4,
    max_iters: int = 5000,
    log_period: int = 20,
    device: str = "cpu",
):
    """
    Train the detector. We use SGD with momentum and step decay.
    """

    detector.to(device=device)

    # Optimizer: use SGD with momentum.
    # Use SGD with momentum:
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, detector.parameters()),
        momentum=0.9,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    # LR scheduler: use step decay at 70% and 90% of training iters.
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(0.6 * max_iters), int(0.9 * max_iters)]
    )

    # Keep track of training loss for plotting.
    loss_history = []

    train_loader = infinite_loader(train_loader)
    detector.train()

    for _iter in range(max_iters):
        # Ignore first arg (image path) during training.
        _, images, gt_boxes = next(train_loader)

        images = images.to(device)
        gt_boxes = gt_boxes.to(device)

        # Dictionary of loss scalars.
        losses = detector(images, gt_boxes)

        # Ignore keys like "proposals" in RPN.
        losses = {k: v for k, v in losses.items() if "loss" in k}

        optimizer.zero_grad()
        total_loss = sum(losses.values())
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Print losses periodically.
        if _iter % log_period == 0:
            loss_str = f"[Iter {_iter}][loss: {total_loss:.3f}]"
            for key, value in losses.items():
                loss_str += f"[{key}: {value:.3f}]"

            print(loss_str)
            loss_history.append(total_loss.item())

    # Plot training loss.
    plt.title("Training loss history")
    plt.xlabel(f"Iteration (x {log_period})")
    plt.ylabel("Loss")
    plt.plot(loss_history)
    plt.show()


def inference_with_detector(
    detector,
    test_loader,
    idx_to_class,
    score_thresh: float,
    nms_thresh: float,
    output_dir: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    device:str = "cpu",
):

    # ship model to GPU
    detector.to(dtype=dtype, device=device)

    detector.eval()
    start_t = time.time()

    # Define an "inverse" transform for the image that un-normalizes by ImageNet
    # color. Without this, the images will NOT be visually understandable.
    inverse_norm = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
            transforms.Normalize(
                mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]
            ),
        ]
    )

    if output_dir is not None:
        det_dir = "mAP/input/detection-results"
        gt_dir = "mAP/input/ground-truth"
        if os.path.exists(det_dir):
            shutil.rmtree(det_dir)
        os.mkdir(det_dir)
        if os.path.exists(gt_dir):
            shutil.rmtree(gt_dir)
        os.mkdir(gt_dir)

    for iter_num, test_batch in enumerate(test_loader):
        image_paths, images, gt_boxes = test_batch
        images = images.to(dtype=dtype, device=device)

        with torch.no_grad():
            if score_thresh is not None and nms_thresh is not None:
                # shapes: (num_preds, 4) (num_preds, ) (num_preds, )
                pred_boxes, pred_classes, pred_scores = detector(
                    images,
                    test_score_thresh=score_thresh,
                    test_nms_thresh=nms_thresh,
                )

        # Skip current iteration if no predictions were found.
        if pred_boxes.shape[0] == 0:
            continue

        # Remove padding (-1) and batch dimension from predicted / GT boxes
        # and transfer to CPU. Indexing `[0]` here removes batch dimension:
        gt_boxes = gt_boxes[0]
        valid_gt = gt_boxes[:, 4] != -1
        gt_boxes = gt_boxes[valid_gt].cpu()

        valid_pred = pred_classes != -1
        pred_boxes = pred_boxes[valid_pred].cpu()
        pred_classes = pred_classes[valid_pred].cpu()
        pred_scores = pred_scores[valid_pred].cpu()

        image_path = image_paths[0]
        # Un-normalize image tensor for visualization.
        image = inverse_norm(images[0]).cpu()

        # Combine predicted classes and scores into boxes for evaluation
        # and visualization.
        pred_boxes = torch.cat(
            [pred_boxes, pred_classes.unsqueeze(1), pred_scores.unsqueeze(1)], dim=1
        )

        # write results to file for evaluation (use mAP API https://github.com/Cartucho/mAP for now...)
        if output_dir is not None:
            file_name = os.path.basename(image_path).replace(".jpg", ".txt")
            with open(os.path.join(det_dir, file_name), "w") as f_det, open(
                os.path.join(gt_dir, file_name), "w"
            ) as f_gt:
                for b in gt_boxes:
                    f_gt.write(
                        f"{idx_to_class[b[4].item()]} {b[0]:.2f} {b[1]:.2f} {b[2]:.2f} {b[3]:.2f}\n"
                    )
                for b in pred_boxes:
                    f_det.write(
                        f"{idx_to_class[b[4].item()]} {b[5]:.6f} {b[0]:.2f} {b[1]:.2f} {b[2]:.2f} {b[3]:.2f}\n"
                    )
        else:
            eecs598.utils.detection_visualizer(
                image, idx_to_class, gt_boxes, pred_boxes
            )

    end_t = time.time()
    print(f"Total inference time: {end_t-start_t:.1f}s")
