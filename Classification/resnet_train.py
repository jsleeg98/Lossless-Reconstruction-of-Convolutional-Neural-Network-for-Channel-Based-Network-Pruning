import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import random
import numpy as np

# reproduce
def reproduce(seed):
    seed = seed
    deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('-seed', type=int, default=777)
parser.add_argument('-m', '--model', type=str, default='resnet50')
parser.add_argument('-tb', type=str, default='test')
parser.add_argument('-g', '--gpu', type=str, default='cuda:0')
parser.add_argument('-lr', type=float, default=0.1)
parser.add_argument('-gamma', type=float, default=0.1)
parser.add_argument('-wd', type=float, default=0.0001)
parser.add_argument('-result_dir', type=str, default='train_result')
parser.add_argument('-dataset_dir', type=str, default='./datasets/')

args = parser.parse_args()

writer = SummaryWriter(f'logs/classification/{args.tb}')

for name, value in vars(args).items():
    print(f'{name} : {value}')
    writer.add_text(f'{name}', f'{value}')

reproduce(args.seed)  # setting random seed


# make dir for saving results
os.makedirs(f'./{args.result_dir}/{args.tb}', exist_ok=True)
os.makedirs(f'./{args.dataset_dir}', exist_ok=True)

# train dataset
# data augmentation
# data preprocessing
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
])

# test dataset
# data preprocessing
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
])

trainset = torchvision.datasets.CIFAR10(f'./{args.dataset_dir}', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=128,
                                          shuffle=True,
                                          drop_last=True,
                                          pin_memory=True,
                                          num_workers=8
                                          )

testset = torchvision.datasets.CIFAR10(f'./{args.dataset_dir}', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=128,
                                         shuffle=False,
                                         drop_last=True,
                                         pin_memory=True,
                                         num_workers=8
                                         )

if args.model == 'resnet18':
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.conv1 = conv1
    model.fc = nn.Linear(512, 10)  # CIFAR10
    model = nn.DataParallel(model)
elif args.model == 'resnet34':
    model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
    conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.conv1 = conv1
    model.fc = nn.Linear(512, 10)  # CIFAR10
    model = nn.DataParallel(model)
elif args.model == 'resnet50':
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.conv1 = conv1
    model.fc = nn.Linear(2048, 10)  # CIFAR10
    model = nn.DataParallel(model)
elif args.model == 'resnet101':
    model = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V2)
    conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.conv1 = conv1
    model.fc = nn.Linear(2048, 10)  # CIFAR10
    model = nn.DataParallel(model)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=0.9, weight_decay=args.wd)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=args.gamma)


device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
print(f'device : {device}')
model.to(device)

best_acc = 0
for epoch in range(100):  # loop over the dataset multiple times
    model.train()
    running_loss = 0.0
    print(f'epoch : {epoch + 1}')
    writer.add_scalar("LR/train", optimizer.param_groups[0]['lr'], epoch)
    for data in tqdm(trainloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss
        loss.backward()
        optimizer.step()
    scheduler.step()
    writer.add_scalar("Loss/train", running_loss, epoch)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f'Accuracy : {100 * correct / total}%')
    writer.add_scalar("Acc/test", acc, epoch)

    if best_acc < acc:
        torch.save(model.module.state_dict(), f'./{args.result_dir}/{args.tb}.pth')
        best_acc = acc
        print('save model')
    writer.flush()

print(f'best acc : {best_acc}')
writer.add_text('best acc', str(best_acc))
writer.close()



