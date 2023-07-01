import random

import torch
from torchvision.transforms import transforms, functional

transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()


def make_pattern(input_shape, mask_value, device, pattern_tensor, x_top, y_top):
    print("input_shape:")
    print(input_shape)
    full_image = torch.zeros(input_shape)
    # full_image = full_image.unsqueeze_(0)
    full_image.fill_(mask_value)

    x_bot = x_top + pattern_tensor.shape[0]
    y_bot = y_top + pattern_tensor.shape[1]

    if x_bot >= input_shape[0] or \
            y_bot >= input_shape[1]:
        raise ValueError(f'Position of backdoor outside image limits:'
                         f'image: {input_shape}, but backdoor'
                         f'ends at ({x_bot}, {y_bot})')

    if len(input_shape) == 2:
        full_image[x_top:x_bot, y_top:y_bot] = pattern_tensor

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        full_image = full_image.unsqueeze_(0)

        # transform = transforms.Compose([
        #     transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        # ])  # 修改的位置

        mask = 1 * (full_image != mask_value).to(device)

        pattern = normalize(full_image).to(device)
    elif len(input_shape) == 3:
        for i in range(3):
            full_image[x_top:x_bot, y_top:y_bot, i] = pattern_tensor

            # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                                  std=[0.229, 0.224, 0.225])
            normalize = transforms.Normalize((0.1307,), (0.3081,))

            mask = 1 * (full_image != mask_value).to(device)

            pattern = normalize(full_image).to(device)

    return mask, pattern

# def synthesize_inputs(pattern, mask, batch, attack_portion=None):
#     batch.data[:attack_portion] = (1 - mask) * batch.data[:attack_portion] + \
#                                   mask * pattern
#
#     return
#
#
# def synthesize_labels(batch, backdoor_label, attack_portion=None):
#     batch.target[:attack_portion].fill_(backdoor_label)
#
#     return
