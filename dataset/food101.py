import torch.utils.data as data
from PIL import Image

def read_list(root, fileList):
    imgList = []
    c_dict = []
    k=-1
    with open(root + '/'+ fileList, 'r') as file:
        for line in file.readlines():

            label_name, _ = line.strip().split('/')
            imgP = line.strip()

            if not label_name in c_dict:
                k += 1
                c_dict.append(label_name)
                label = k

            else:
                label = k
            imgPath = root + '/images/' + imgP + '.jpg'
            imgList.append((imgPath, int(label)))

    return imgList

class Food101(data.Dataset):
    def __init__(self, root, transform):
        self.imgList = read_list(root, 'meta/test.txt')
        self.transform = transform

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img = Image.open(imgPath)
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgList)
