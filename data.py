import numpy as np
from torchvision import transforms
from torchvision.transforms import ToTensor, ToPILImage, Compose
import torch.utils.data.dataset as dataset
from PIL import Image
import os
import cv2

label_path = "/home/zhangheng/PycharmProjects/Res_DN/LHR/"
transform = transforms.Compose([transforms.ToTensor()])


class My_dataset(dataset.Dataset):
    def __init__(self, filenames, transform=None):
        super(My_dataset, self).__init__()
        self.filenames = filenames
        self.transform = transform
        img = [os.path.join(filenames, image) for image in os.listdir(filenames)]
        img = sorted(img, key=lambda x: (x.split("/")[-1].split(".")[-2]))
        # img_num=len(img)
        # print(img)
        self.img = img

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img_path = self.img[idx]
        # print(img_path)

        # modified start
        # image = Image.open(img_path)
        # image = ToTensor()(image)
        # modified end

        # image = cv2.imread(img_path)
        image = Image.open(img_path)
        image = ToTensor()(image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image).astype(np.float32)
        # print(image)
        # image = image.transpose((2, 0, 1))
        a = img_path.split("/")[-1]
        # print(a)
        self.label_path = os.path.join(label_path, a)
        # print(self.label_path)
        # labels=Image.open(self.label_path).convert('RGB')
        # labels = cv2.imread(self.label_path)
        labels = Image.open(self.label_path)
        labels = ToTensor()(labels)
        # labels = cv2.cvtColor(labels, cv2.COLOR_BGR2RGB)
        labels = np.array(labels).astype(np.float32)
        # labels = labels.transpose((2, 0, 1))
        return image, labels
