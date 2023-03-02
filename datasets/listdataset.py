import torch.utils.data as data


class ListDataset(data.Dataset):
    def __init__(self, root, dataset, path_list, image_transform=None, label_transform=None,
                 co_transform=None, loader=None, datatype=None):

        self.root = root
        self.dataset = dataset
        self.img_path_list = path_list
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.co_transform = co_transform
        self.loader = loader
        self.datatype = datatype

    def __getitem__(self, index):
        img_path = self.img_path_list[index][:-1]
        # We do not consider other datasets in this work
        assert self.dataset == 'bsd500'
        assert (self.image_transform is not None) and (self.label_transform is not None)

        inputs, label = self.loader(img_path, img_path.replace('_img.jpg', '_label.png'))

        if self.co_transform is not None:
            inputs, label = self.co_transform([inputs], label)

        if self.image_transform is not None:
            image = self.image_transform(inputs[0])

        if self.label_transform is not None:
            label = self.label_transform(label)

        return image, label

    def __len__(self):
        return len(self.img_path_list)
