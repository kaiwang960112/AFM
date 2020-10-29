import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
import os
import torch.nn.parallel
import argparse
import afm
from dataset import Food101, Food101N

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('--network', default='resnet50', type=str,
                   choices=['resnet50', 'resnet101', 'resnet152'],
                   help='model architecture')
parser.add_argument('--train-batch', default=256, type=int, help='train batchsize')
parser.add_argument('--test-batch', default=200, type=int, help='test batchsize')
parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[15, 25], help='decrease learning rate at these epochs')
parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--checkpoint', default='./checkpoint', type=str, help='path to save checkpoint')
parser.add_argument('--gamma', default=0.1, type=float, help='LR is multiplied by gamma on schedule')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--weight-naive', default=1.5, type=float, help='loss weight of naive classifier')
parser.add_argument('--weight-afm', default=0.5, type=float, help='loss weight of afm')
parser.add_argument('--dataset', default='food', type=str, help='dataset')
parser.add_argument('--data-root', default='./data/food', type=str, help='data root')
parser.add_argument('--device_ids', default='0,1,2,3,4,5,6,7', type=str,
                    help='number of CUDA_VISIBLE_DEVICES')

def mkdir(s):
    os.system("mkdir -p %s"%s)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
    best_acc1 = 0.
    best_acc2 = 0.
    best_epoch = 0

    save_path = os.path.join('results/food/')
    mkdir(save_path)
    save_path_result = os.path.join('results/food_result')
    mkdir(save_path_result)

    if args.dataset == 'food':
        listfile = './data/Food-101N_release/meta/imagelist.tsv'
        val_listfile = './data/food-101/meta/test.txt'

        num_cls = 101
        train_loader = torch.utils.data.DataLoader(
            Food101N(root=args.data_root + '/Food-101N_release',
                     transform=transforms.Compose([
                         transforms.RandomResizedCrop(224),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                     ])),
            batch_size=args.train_batch, shuffle=True,
            num_workers=args.workers, pin_memory=True,drop_last=True)

        valid_loader = torch.utils.data.DataLoader(
            Food101(root=args.data_root + '/food-101',
                    transform=transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])),
            batch_size=args.test_batch, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    model = afm.__dict__[args.network](pretrained=True, num_classes=num_cls)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)

    ## training
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)
        running_loss = 0.0
        running_corrects = 0
        corrects_all = 0.0
        train_num_total = 0

        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()

        model.train()
        for i, (inputs, labels) in enumerate(train_loader):

            data_time.update(time.time() - end)
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            scores, scores_naive = model(inputs)

            _, preds = torch.max(scores_naive.data, 1)
            loss = args.weight_naive * criterion(scores, labels) + args.weight_afm * criterion(scores_naive, labels)

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()
            # zero the parameter gradients
            # statistics
            running_loss += loss.item()
            num_correct = torch.sum(preds == labels.data)
            corrects_all += num_correct.float()
            train_num_total += len(labels.data)
            running_corrects = num_correct.float() / float(preds.shape[0])
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print('{} Epoch, {} Iter,Time :{:.3f},Data:{:.3f} Loss: {:.4f}, Acc: {:.4f}'.format(
                    epoch, i, batch_time.avg, data_time.avg, loss.item(), running_corrects))

        running_loss /= float(len(train_loader))
        corrects_all /= float(train_num_total)

        print('Train Loss: {:.4f}, Acc: {:.4f}'.format(running_loss,
                                                       corrects_all))

        model.eval()
        running_valid_loss = 0.0
        running_valid_corrects = 0
        valid_corrects_all = 0.0
        valid_num_total = 0
        end = time.time()
        for i, (inputs, labels) in enumerate(valid_loader):

            batch_time = AverageMeter()
            inputs = inputs.cuda()
            labels = labels.cuda()

            with torch.no_grad():

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

            running_valid_loss += loss.item()
            num_valid_correct = torch.sum(preds == labels.data)
            valid_corrects_all += num_valid_correct.float()
            valid_num_total += len(labels.data)
            running_valid_corrects = num_valid_correct.float() / float(preds.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 20 == 0:
                print('{} Epoch, {} Iter, Time: {:.3f}, Loss: {:.4f} Acc: {:.4f}'.format(
                    epoch, i, batch_time.avg, loss.item(), running_valid_corrects))

        running_valid_loss /= float(len(valid_loader))
        valid_corrects_all /= float(valid_num_total)
        print('Valid Loss: {:.6f}, Acc: {:.6f}'.format(running_valid_loss,
                                                       valid_corrects_all))

        torch.save(model.state_dict(), save_path + 'model_%d.pkl'%epoch)
        lr_scheduler.step()
        is_best = valid_corrects_all > best_acc1
        if is_best:
            best_acc1 = valid_corrects_all
            best_epoch = epoch
            best_model = model
            print('Best best_acc1 {}'.format(best_acc1))
            torch.save(best_model.state_dict(), save_path + 'model_best.pkl')
    save_obj.close()
