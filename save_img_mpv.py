import os
import shutil

import torch
import torchvision.utils as vutils


def save_img(print_list, name, index, results_dir, config_path, first=False):
    # pdb.set_trace()
    nrow = len(print_list)
    img = torch.cat(print_list, dim=3)

    directory = os.path.join(results_dir, name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if first:
        shutil.copy(
            config_path, os.path.join(directory, "config.yaml")
        )  # copy config file to output folder
    else:
        pass
    path = os.path.join(directory, "{}".format(index) + ".jpg")
    img = img.permute(1, 0, 2, 3).contiguous()
    vutils.save_image(
        img.view(1, img.size(0), -1, img.size(3)).data,
        path,
        nrow=nrow,
        padding=0,
        normalize=True,
    )