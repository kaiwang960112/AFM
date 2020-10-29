import torch.utils.data as data
from PIL import Image

def read_list(root, fileList):
    imgList = []
    c_dict = []
    k=-1
    with open(root + '/' + fileList, 'r') as file:
        for line in file.readlines():
            row = line.strip().split('\t')
            if len(row) == 1:
                label_name, _ = row[0].strip().split('/')
                imgP = line.strip()

                if label_name == 'class_name':
                    continue
                else:
                    if not label_name in c_dict:
                        k += 1
                        c_dict.append(label_name)
                        label = k

                    else:
                        label = k
                    imgPath = root + '/images/' + imgP
                    imgList.append((imgPath, int(label)))
            else:
                imgP = row[0]
                label_name, _ = row[0].strip().split('/')

                if label_name == 'class_name':
                    continue
                else:
                    if not label_name in c_dict:
                        k += 1
                        c_dict.append(label_name)
                        label = k

                    else:
                        label = k
                    imgPath = root + '/images/' + imgP
                    imgList.append((imgPath, int(label)))

    return imgList

class Food101N(data.Dataset):
    def __init__(self, root, transform):
        self.imgList = read_list(root, 'meta/imagelist.tsv')
        self.transform = transform

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img = Image.open(imgPath)
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgList)
