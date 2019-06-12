import torch
import numpy as np
import cv2
from torch.utils.data import Dataset



import glob
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
class CelebA(Dataset):
    def __init__(self,
                 root,
                 train=True,
                 size=256,
                 augmentation=None):
        self.root = root
        self.train = train
        # self.joint_transform = joint_transform
        files_path = glob.glob(root + 'masks/*bmp')
        files_names = [x.replace(root+'masks/','') for x in files_path]
        train_limit = (9 * len(files_names)) // 10
        self.files_names = files_names[train_limit:]
        self.augmentation = augmentation
        self.mask_transforms =transforms.Compose([  
            transforms.Resize(size=(size, size)),
            transforms.ToTensor(),

        ])
        self.img_transforms = transforms.Compose([  
            transforms.Resize(size=(size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        if train:
            self.files_names = files_names[0:train_limit]
        

        print(len(self.files_names))
        
    def __len__(self):
        return len(self.files_names)

    def __getitem__(self, index):
        file_name = self.files_names[index]
        _img = np.array(Image.open(self.root + 'images/' + file_name.replace('bmp', 'png')).convert('RGB'))
        mask_bmp = Image.open(self.root + 'masks/' + file_name)
        mask_array = np.array(mask_bmp)
        mask_array = (mask_array == 128.0).astype(float)

        if self.augmentation is not None:
            data = {"image": _img, "mask": mask_array}
            augmented = self.augmentation(**data)
            _img, mask_array = augmented["image"], augmented["mask"]


        _mask = Image.fromarray(mask_array)
        _img = Image.fromarray(_img)

        _img, _mask = self.img_transforms(_img), self.mask_transforms(_mask)
        
        return _img, _mask


# class RoboticsDataset(Dataset):
#     def __init__(self, file_names, to_augment=False, transform=None, mode='train', problem_type=None):
#         self.file_names = file_names
#         self.to_augment = to_augment
#         self.transform = transform
#         self.mode = mode
#         self.problem_type = problem_type

#     def __len__(self):
#         return len(self.file_names)

#     def __getitem__(self, idx):
#         img_file_name = self.file_names[idx]
#         image = load_image(img_file_name)
#         mask = load_mask(img_file_name, self.problem_type)

#         data = {"image": image, "mask": mask}
#         augmented = self.transform(**data)
#         image, mask = augmented["image"], augmented["mask"]

#         if self.mode == 'train':
#             if self.problem_type == 'binary':
#                 return img_to_tensor(image), torch.from_numpy(np.expand_dims(mask, 0)).float()
#             else:
#                 return img_to_tensor(image), torch.from_numpy(mask).long()
#         else:
#             return img_to_tensor(image), str(img_file_name)


# def load_image(path):
#     img = cv2.imread(str(path))
#     return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# def load_mask(path, problem_type):
#     if problem_type == 'binary':
#         mask_folder = 'binary_masks'
#         factor = prepare_data.binary_factor
#     elif problem_type == 'parts':
#         mask_folder = 'parts_masks'
#         factor = prepare_data.parts_factor
#     elif problem_type == 'instruments':
#         factor = prepare_data.instrument_factor
#         mask_folder = 'instruments_masks'

#     mask = cv2.imread(str(path).replace('images', mask_folder).replace('jpg', 'png'), 0)

#     return (mask / factor).astype(np.uint8)
