from os.path import exists, join, splitext
import numpy as np
from PIL import Image
import utils.flowlib as fl
from torch.utils.data import Dataset
from . import flowtransforms as transforms
import json 
from pycocotools.coco import COCO

def generate_coco_info(json_fn):
    coco = COCO(json_fn)
    with open(json_fn) as f:
        sur_dir = json.load(f)
    
    img_dir_list = sur_dir['images']
    reverse_img_dir = {img_dir['file_name']:img_dir['id'] for img_dir in img_dir_list}

    return coco, reverse_img_dir

class BDD_Data(Dataset):
    # Disparity annotations are transformed into flow format;
    # Sparse annotations possess an extra dimension as the valid mask;
    def __init__(self,
                 mode,
                 data_root,
                 data_list,
                 coco_file,
                 reverse_img_dir,
                 transform=None,
                 out_size=False):
        assert mode in ["flow", "stereo"]
        self.mode = mode
        self.data_root = data_root
        self.data_list = self.read_lists(data_list)
        self.transform = transform
        self.out_size = out_size
        self.coco = coco_file
        self.reverse_img_dir = reverse_img_dir
        # self.label_num = 0

    def __len__(self):
        return len(self.data_list)

    def generate_instance_maps(self, img_id, img_id_des):
        sur_map = np.zeros(1280, 720)
        tar_map = np.zeros(1280, 720)
        annIds = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        annIds_des = self.coco.getAnnIds(imgIds=[img_id_des], iscrowd=None)
        annos = self.coco.loadAnns(annIds)
        annos_des = self.coco.loadAnns(annIds_des)
        instance_ids = [anno['instance_id'] for anno in annos]
        instance_ids_des = [anno['instance_id'] for anno in annos_des]
        curr_instance_id = 0
        for anno, instance_id in zip(annos, instance_ids):
            if instance_id not in instance_ids_des:
                continue

            idx = instance_ids_des.index(instance_id)
            anno_des = annos_des[idx]
            mask = coco.annToMask(anno)
            mask_des = coco.annToMask(anno_des)
            sur_map[np.where(mask>0)] = curr_instance_id
            tar_map[np.where(mask_des>0)] = curr_instance_id
            curr_instance_id += 1
        
        sur_map = np.expand_dims(sur_map, axis=-1)
        tar_map = np.expand_dims(tar_map, axis=-1)
        return sur_map, tar_map

    def __getitem__(self, index):
        # img_num = len(self.data_list[index])
        img_list = []
        label_list = []
        img_list.append(read_gen(join(self.data_root, self.data_list[index][0]), "image"))
        img_list.append(read_gen(join(self.data_root, self.data_list[index][1]), "image"))
        img_id = self.reverse_img_dir[self.data_list[index][1]]
        img_id_des = self.reverse_img_dir[self.data_list[index][0]]
        sur_map, tar_map = self.generate_instance_maps(img_id, img_id_des)
        label_list.append(sur_map)
        label_list.append(tar_map)

        data = [img_list, label_list]
        data = list(self.transform(*data))

        if self.out_size:
            data.append(np.array(img_list[0].size, dtype=int))

        return tuple(data)

    def read_lists(self, data_list):
        assert exists(data_list)
        samples = [line.strip().split(' ') for line in open(data_list, 'r')]
        return samples


def read_gen(file_name, mode):
    ext = splitext(file_name)[-1]
    if mode == 'image':
        assert ext in ['.png', '.jpeg', '.ppm', '.jpg']
        return Image.open(file_name)
    elif mode == 'flow':
        assert ext in ['.flo', '.png', '.pfm']
        return fl.read_flow(file_name)
    elif mode == 'stereo':
        assert ext in ['.png', '.pfm']
        return fl.read_disp(file_name)
    else:
        raise ValueError('Unknown mode {}'.format(mode))


def get_transform(dataset_name, task, evaluate=True):
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    pad_mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]

    val_transform = None
    if dataset_name == 'FlyingChairs':
        train_transform = transforms.Compose([
            transforms.RandomScale([1, 2]),
            transforms.Crop([384, 512], 'rand', pad_mean),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        if evaluate:
            val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    elif dataset_name == 'FlyingThings3D':
        if task == 'flow':
            train_transform = transforms.Compose([
                transforms.Crop([384, 832], 'rand', pad_mean),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

            if evaluate:
                val_transform = transforms.Compose([
                    transforms.Crop([384, 832], 'center', pad_mean),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
        else:
            train_transform = transforms.Compose([
                transforms.Crop([320, 896], 'rand', pad_mean),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

            if evaluate:
                val_transform = transforms.Compose([
                    transforms.Crop([320, 896], 'center', pad_mean),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])

    elif dataset_name == 'KITTI':
        if task == 'flow':
            train_transform = transforms.Compose([
                transforms.MultiScaleRandomCrop([0.5, 1.15], [320, 896],
                                                'nearest'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.RandomPhotometric(
                    noise_stddev=0.0,
                    min_contrast=-0.3,
                    max_contrast=0.3,
                    brightness_stddev=0.02,
                    min_color=0.9,
                    max_color=1.1,
                    min_gamma=0.7,
                    max_gamma=1.5),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            train_transform = transforms.Compose([
                transforms.MultiScaleRandomCrop([0.5, 1.15], [320, 896],
                                                'nearest'),
                transforms.ToTensor(),
                transforms.RandomPhotometric(
                    noise_stddev=0.0,
                    min_contrast=-0.3,
                    max_contrast=0.3,
                    brightness_stddev=0.02,
                    min_color=0.9,
                    max_color=1.1,
                    min_gamma=0.7,
                    max_gamma=1.5),
                transforms.Normalize(mean=mean, std=std)
            ])

        if evaluate:
            val_transform = transforms.Compose([
                transforms.Resize([1280, 384], 'nearest'),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    elif dataset_name == 'MPISintel':
        train_transform = transforms.Compose([
            transforms.MultiScaleRandomCrop([0.5, 1.13], [384, 768],
                                            'bilinear'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomPhotometric(
                noise_stddev=0.0,
                min_contrast=-0.3,
                max_contrast=0.3,
                brightness_stddev=0.02,
                min_color=0.9,
                max_color=1.1,
                min_gamma=0.7,
                max_gamma=1.5),
            transforms.Normalize(mean=mean, std=std)
        ])

        if evaluate:
            val_transform = transforms.Compose([
                transforms.Resize([1024, 448], 'bilinear'),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    else:
        raise ValueError('Unknown dataset name {}'.format(dataset_name))

    return train_transform, val_transform
