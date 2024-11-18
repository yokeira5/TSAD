from sympy.stats.sampling.sample_numpy import numpy
from torch.utils.data import Dataset
from visdom.utils.server_utils import window

from .fag import *
from .stitch import *
import os
from glob import glob

import torch
import torch.utils.data
from PIL import Image, ImageOps
from torchvision import transforms

from skimage.feature import local_binary_pattern


MVTec_DEFECT_TYPE = ['defect-free', 'defect']

# max dataset size: 391(hazelnut)
# dataset size = 13 *32 = 416

default_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

fileswith = ('.png', '.jpg', '.jpeg', '.bmp')
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]


def get_image_file_path(base_dir, img_fname):
    possible_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    for ext in possible_extensions:
        fpath = os.path.join(base_dir, img_fname + ext)
        if os.path.exists(fpath):
            return fpath
    return None

class MVTec(Dataset):
    def __init__(self, dataset_path='./Data/MVTec', class_name='bottle', is_train=True, load_size=256, numbers = 100, transform_train=None):
        super().__init__()
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.load_size = load_size
        self.numbers = numbers

        # load data path
        if is_train:
            transform_train = FAG(load_size=load_size, is_findobj=0)
            self.images, self.labels, self.masks = self.load_train_folder()
        else:
            self.images, self.labels, self.masks = self.load_test_folder()

        if transform_train is not None:
            self.transform_train = transform_train
        else:
            self.transform_train = transforms.Compose([transforms.ToTensor()])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
        self.transform_mask = transforms.Compose([transforms.ToTensor()])
    
    def __getitem__(self, index):
        img_path, label, mask = self.images[index], self.labels[index], self.masks[index]
        # ../defect_type/000.png
        # img_name = img.split('/')[-2] + '_'
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img = Image.open(img_path).convert('RGB')
        if self.is_train:
            # data augmentation on train data
            img, aug, aug_mask, aug_noise = self.transform_train(img)
            return img, aug, label, aug_mask, aug_noise
        else:
            img = self.transform_test(img)
            if MVTec_DEFECT_TYPE[label] == 'defect-free':
                mask = torch.zeros([1, self.load_size, self.load_size])
            else:
                mask = Image.open(mask).convert('L')
                mask = self.transform_mask(mask)
            return img, mask, label, img_name
    
    def __len__(self):
        return len(self.images)

    def load_train_folder(self):
        images, labels, masks = [], [], []
        img_dir = os.path.join(self.dataset_path, self.class_name, "TRAIN", "defect-free")[:self.numbers]
        img_fpath_list = sorted([os.path.join(img_dir, f)
                                     for f in os.listdir(img_dir)
                                     if f.endswith(fileswith)])
        images = img_fpath_list
        labels.extend([0] * len(images))
        masks.extend([None] * len(images))
        assert len(images) == len(labels), 'number of x and y should be same'
        return list(images), list(labels), list(masks)

    def load_test_folder(self):
        images, labels, masks = [], [], []
        img_dir = os.path.join(self.dataset_path, self.class_name, "TEST")
        gt_dir = os.path.join(self.dataset_path, self.class_name, "TEST", 'groundtruth')

        img_types = MVTec_DEFECT_TYPE

        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue

            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith(fileswith)])

            if img_type == 'defect-free':
                img_fpath_list = img_fpath_list

            images.extend(img_fpath_list)

            # load gt labels
            if img_type == 'defect-free':
                labels.extend([0] * len(img_fpath_list))
                masks.extend([None] * len(img_fpath_list))
            else:
                labels.extend([1] * len(img_fpath_list))
                gt_type_dir = gt_dir #os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [get_image_file_path(gt_type_dir, img_fname) for img_fname in img_fname_list]
                masks.extend(gt_fpath_list)

        assert len(images) == len(labels), 'number of x and y should be same'

        return list(images), list(labels), list(masks)


class ST(Dataset):
    def __init__(self, dataset_path='./Data/MVTec', target_list = [], is_train=True, load_size=256, numbers = 100, transform_train=None):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((load_size, load_size)),
                transforms.ToTensor(),  # Scales data into [0,1]
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((load_size, load_size)),
                transforms.ToTensor(),  # Scales data into [0,1] 
            ]
        )
        if is_train:
            transform_train = FAG(load_size=load_size, is_findobj=0)

            image_files = glob(
                os.path.join(dataset_path , "*.jpg")
            )

            self.image_files = image_files[:1000]

        else:
            self.image_files = []
            image_files = []
            for cur_data in target_list:
                for extension in ["*.jpg", "*.jpeg", "*.png"]:
                    image_files.extend(glob(os.path.join(dataset_path, cur_data, extension))[: 500])

            self.image_files = image_files

        self.is_train = is_train
        
        if transform_train is not None:
            self.transform_train = transform_train
        else:
            self.transform_train = transforms.Compose([transforms.ToTensor()])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((load_size, load_size)),
        ])
        self.transform_mask = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file).convert('RGB')

        img_name = os.path.splitext(os.path.basename(image_file))[0]

        if self.is_train:
            image, aug, aug_mask, aug_noise = self.transform_train(image)

            return aug, aug_mask, img_name
        else:
            folder_name = os.path.basename(os.path.dirname(image_file))

            # radius = 1  # 鄰域的半徑
            # n_points = 8 * radius  # 周圍像素點的數量
            # img_gray_array = local_binary_pattern(np.array(image), n_points, radius, method='uniform')
            # image = Image.fromarray(img_gray_array, 'L')

            image = self.image_transform(image)

            mask = torch.zeros([1, image.shape[-2], image.shape[-1]])

            return image, mask, folder_name, img_name

    def __len__(self):
        return len(self.image_files)


class ST_double(Dataset):
    def __init__(self, dataset_path='./Data/MVTec', target_list=[], is_train=True, load_size=256, numbers=100,
                 transform_train=None):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((load_size, load_size)),
                transforms.ToTensor(),  # Scales data into [0,1]
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((load_size, load_size)),
                transforms.ToTensor(),  # Scales data into [0,1]
            ]
        )
        if is_train:
            transform_train = FAG_double(load_size=load_size, is_findobj=0)

            image_files = glob(
                os.path.join(dataset_path, "*.jpg")
            )

            image_files = [(image_files[i], image_files[i + 1]) for i in range(0, len(image_files) - 1, 1)]

            self.image_files = image_files[:1000]

        else:
            self.image_files = []
            image_files = []
            for cur_data in target_list:
                for extension in ["*.jpg", "*.jpeg", "*.png"]:
                    files = glob(os.path.join(dataset_path, cur_data, extension))[:500]
                    image_files.extend([(files[i], files[i + 1]) for i in range(0, len(files) - 1, 1)])


            self.image_files = image_files

        self.is_train = is_train

        if transform_train is not None:
            self.transform_train = transform_train
        else:
            self.transform_train = transforms.Compose([transforms.ToTensor()])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((load_size, load_size)),
        ])
        self.transform_mask = transforms.Compose([transforms.ToTensor()])

    def get_window_size(self, left_img, right_img, blending_mode = "linearBlendingWithConstant"):
        # blending_mode = "linearBlendingWithConstant"  # three mode - noBlending、linearBlending、linearBlendingWithConstant
        # left_img
        width_small, _ = left_img.size

        left_img = np.array(left_img)
        right_img = np.array(right_img)

        stitcher = Stitcher()
        warp_img = stitcher.stitch([left_img, right_img], blending_mode)

        _, width_large = warp_img.shape[:2]

        window_size = random.randint(15, width_large - width_small)

        right_img = warp_img[:, window_size: window_size + width_small]

        right_img = Image.fromarray(right_img)

        return window_size, right_img

    def __getitem__(self, index):
        # 取出圖片對
        image_file1, image_file2 = self.image_files[index]

        # 打開圖片
        image1 = Image.open(image_file1).convert('RGB')
        image2 = Image.open(image_file2).convert('RGB')

        img_name = os.path.splitext(os.path.basename(image_file2))[0]

        if self.is_train:
            window_size = 20
            # window_size, image2 = self.get_window_size(image1, image2)
            image, aug, aug2, aug_mask, aug_noise = self.transform_train(image1, image2, window_size)

            concatenated = torch.cat((aug, aug2), dim=0)

            return image, concatenated, aug_mask, img_name, window_size

        else:
            folder_name = os.path.basename(os.path.dirname(image_file2))

            # radius = 1  # 鄰域的半徑
            # n_points = 8 * radius  # 周圍像素點的數量
            # img_gray_array = local_binary_pattern(np.array(image), n_points, radius, method='uniform')
            # image = Image.fromarray(img_gray_array, 'L')

            image1 = self.image_transform(image1)
            image2 = self.image_transform(image2)

            concatenated = torch.cat((image1, image2), dim=0)

            mask = torch.zeros([1, image2.shape[-2], image2.shape[-1]])

            return concatenated, mask, folder_name, img_name

    def __len__(self):
        return len(self.image_files)



class Simulate(Dataset):
    def __init__(self, dataset_path='./Data/MVTec', class_name='bottle', is_train=True, load_size=256, numbers=100,
                 transform_train=None):
        super().__init__()
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.load_size = load_size
        self.numbers = numbers

        # load data path
        self.images = self.load_train_folder()

        self.transform_train = transforms.Compose([transforms.ToTensor()])
        self.transform_test = transforms.Compose([transforms.ToTensor()])
        self.transform_mask = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        img_path = self.images[index]
        img = Image.open(img_path).convert('RGB')
        msk = Image.open(img_path.replace('train', 'groundtruth')).convert('L')

        image = self.transform_train(img)
        mask = self.transform_mask(msk)

        return image, mask

    def __len__(self):
        return len(self.images)

    def load_train_folder(self):
        images, labels, masks = [], [], []
        category_dir = os.listdir(os.path.join(self.dataset_path, self.class_name, "train"))
        cue_img_dir = []
        img_fpath_list = []
        for cur_cate in category_dir:
            cue_img_dir = os.path.join(self.dataset_path, self.class_name, "train", cur_cate)
            cur_img_fpath_list = sorted([os.path.join(cue_img_dir, f)
                                         for f in os.listdir(cue_img_dir)
                                         if f.endswith(fileswith)])

            img_fpath_list.extend(cur_img_fpath_list)

        return img_fpath_list


# from torch.utils.data import DataLoader
# test_dataset = Simulate(rf'F:\OUTPUT\DATASET\Simulated\YDFID_1', class_name='CL1')
# test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)
#
# for idx, (img, aug) in enumerate(test_loader):
#     print(img)

