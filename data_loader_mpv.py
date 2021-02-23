import os
from PIL import Image

from torch.utils import data
from torchvision import transforms as T


class AlignedDataset(data.Dataset):
    def __init__(self, dataset, image_dir, transform, mode, data_list=None):
        super(AlignedDataset, self).__init__()
        self.transform = transform
        self.mode = mode
        self.dir_A = os.path.join(image_dir, mode, "image/")
        self.dir_B = os.path.join(image_dir, mode, "cloth/")
        self.data_list = (
            os.path.join(image_dir, mode + "_pairs.txt")
            if data_list is None
            else os.path.join(image_dir, data_list)
        )
        self.A_paths = self.make_dataset(self.dir_A)
        self.B_paths = self.make_dataset(self.dir_B)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        # load data list
        im_paths = []
        c_paths = []
        with open(self.data_list, "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_paths.append(self.dir_A + im_name)
                c_paths.append(self.dir_B + c_name)

        self.im_paths = im_paths
        self.c_paths = c_paths

    def make_dataset(self, dir):
        images = []
        assert os.path.isdir(dir), "%s is not a valid directory" % dir

        for fname in sorted(os.listdir(dir)):
            path = os.path.join(dir, fname)
            images.append(path)

        return images

    def __getitem__(self, index):
        im_path = self.im_paths[index]
        c_path = self.c_paths[index]

        A_img = Image.open(im_path).convert("RGB")
        B_img = Image.open(c_path).convert("RGB")

        A = self.transform(A_img)
        B = self.transform(B_img)

        if self.mode == "train":
            return A, B
        else:
            return A, B, im_path

    def __len__(self):
        return max(self.A_size, self.B_size)


def get_loader(
    image_dir,
    crop_size=216,
    resize=216,
    batch_size=16,
    dataset="MPV_supervised",
    mode="train",
    num_workers=1,
    data_list=None,
):
    """Build and return a data loader."""
    transform = []
    if mode == "train":
        transform.append(T.RandomHorizontalFlip())

    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if data_list is None:
        dataset = AlignedDataset(dataset, image_dir, transform, mode)
    else:
        dataset = AlignedDataset(dataset, image_dir, transform, mode, data_list)

    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        sampler=None,
    )
    return data_loader
