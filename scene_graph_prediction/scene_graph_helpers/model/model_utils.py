from itertools import chain

import timm
from timm.data import resolve_data_config, create_transform


def get_image_model(model_config, only_transforms=False):
    if model_config['IMAGE_MODEL'] == False:
        return None
    image_model = timm.create_model(model_config['IMAGE_MODEL'], pretrained=True, num_classes=0)
    # Freeze the whole model
    for param in image_model.parameters():
        param.requires_grad = False
    # Unfreeze conv head
    for param in chain(image_model.conv_head.parameters()):
        param.requires_grad = True
    image_config = resolve_data_config({}, model=image_model)
    val_image_transform = create_transform(**image_config)
    image_transformations = {'train': val_image_transform, 'val': val_image_transform, 'test': val_image_transform}
    if only_transforms:
        return image_transformations
    return image_model, image_transformations
