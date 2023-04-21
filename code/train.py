import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.utils
import FFA
import model as m
from torch.utils.data import DataLoader
from utils import to_psnr,validation,loss_compute,adjust_learning_rate
from data import TrainData,ValData,ValData_bli,TestData
import os
import math

torch.manual_seed(3)
parser = argparse.ArgumentParser(description='Hyper-parameters for DehazeNet')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-4, type=float)
parser.add_argument('-data_learning_rate', help='Set the learning rate', default=0.05, type=float)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=4, type=int)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-num_epochs', help='the number of the epochs', default=250, type=int)
parser.add_argument('-crop_size', help='Set the crop_size', default=[240, 240], nargs='+', type=int)
parser.add_argument('-val_learning_rate', help='Set the learning rate in the val', default=1e-4, type=float)
args = parser.parse_args()


crop_size = args.crop_size
learning_rate = args.learning_rate
val_learning_rate = args.val_learning_rate
data_learning_rate = args.data_learning_rate
train_batch_size = args.train_batch_size
val_batch_size = args.val_batch_size
num_epochs = args.num_epochs
# train\val\test file directory
train_data_dir = '/home/jxy/projects_dir/fff/dataset/ITS_v2/'
val_data_dir = '/home/jxy/projects_dir/fff/dataset/ITS_v2/'
test_data_dir = '/home/jxy/projects_dir/fff/dataset/SOTS/indoor/'

# get device
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# network
net = FFA.FFA(gps=3, blocks=20)
net = net.to(device)
net_data = m.Data_prob(train_batch_size)
net_data = net_data.to(device)

# --- Build optimizer --- #
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
optimizer_data = torch.optim.Adam(net_data.parameters(), lr=data_learning_rate)


def lr_schedule_cosdecay(t, T=1e6, init_lr=args.learning_rate):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr

# --- Load the network weight --- #
try:
    net.load_state_dict(torch.load('haze_add'.format()))
except:
    pass

try:
    net_data.load_state_dict(torch.load('haze_data'.format()))
except:
    pass

try:
    baseline = torch.load('baseline'.format())
except:
    baseline = 0

# --- Load training data and validation/test data --- #
train_data_loader = DataLoader(TrainData(crop_size,train_data_dir), batch_size=train_batch_size, shuffle=True, num_workers=0,drop_last = True)
val_data_loader = DataLoader(ValData(crop_size,val_data_dir), batch_size=val_batch_size, shuffle=True, num_workers=0,drop_last= True)
val_data_loader_bli = DataLoader(ValData_bli(crop_size,val_data_dir), batch_size=train_batch_size, shuffle=True, num_workers=0,drop_last= True)
test_data_loader = DataLoader(TestData(crop_size,test_data_dir), batch_size=val_batch_size, shuffle=True, num_workers=0,drop_last= True)

# --- Previous PSNR and SSIM in testing --- #
old_val_psnr, old_val_ssim = validation(net, val_data_loader, device)
old_test_psnr, old_test_ssim =[0,0]
print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))

try:
    epoch_s = torch.load('epoch'.format())
except:
    epoch_s = 0

for epoch in range(epoch_s,num_epochs):
    print(1)
    psnr_list = []
    val_psnr_list = []
    start_time = time.time()
    for batch_id, train_data in enumerate(train_data_loader):
        step = batch_id + epoch * (13490//4)
        lr = lr_schedule_cosdecay(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        haze, clear = train_data
        haze = haze.to(device)
        clear = clear.to(device)
        data_input = torch.cat((haze,clear),dim=1)
        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()
        optimizer_data.zero_grad()

        # --- Forward + Backward + Optimize --- #
        net.train()
        data_prob = net_data(data_input)
        add_out = net(haze)
        loss = F.l1_loss(add_out, clear)
        loss.backward()

        # save grad pred
        save_grad_pred = {}
        for name, param in net.named_parameters():
            if param.requires_grad:
                save_grad_pred[name] = param.grad.clone().detach()

        optimizer.zero_grad()
        add_out = net(haze)
        loss_list = loss_compute(data_prob, add_out, clear, train_batch_size)
        loss = sum(loss_list)
        print(f'{batch_id} +  loss: {loss}')
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # --- To calculate average PSNR --- #
        psnr_list.extend(to_psnr(add_out, clear))
        # if not (batch_id % 100):
        #    print('Epoch: {0}, Iteration: {1}'.format(epoch, batch_id))
        loss = 0

        # --- Forward + backward in val --- #
        # --- get the new grad
        for batch_id, train_data in enumerate(val_data_loader_bli):
            haze, clear ,haze_name= train_data
            haze = haze.to(device)
            clear = clear.to(device)
            add_out = net(haze)
            loss = F.l1_loss(add_out, clear)
            loss.backward()

            # save grad in validation
            save_grad_current = {}
            for name, param in net.named_parameters():
                if param.requires_grad:
                    save_grad_current[name] = param.grad.clone().detach()
            break
        # update score
        optimizer.zero_grad()
        r = 0
        optimizer_data.zero_grad()
        for key in save_grad_current:
            reward = (save_grad_current[key] * save_grad_pred[key]).sum()
            r += reward

        if baseline == 0:
            baseline = r
        else:
            baseline = baseline - 0.05 * (baseline - r)

        param_loss = -torch.mean((r.detach() - baseline.detach()) * data_prob.detach() * torch.log(data_prob))

        param_loss.backward()
        optimizer_data.step()

    # --- Calculate the average training PSNR in one epoch --- #
    train_psnr = sum(psnr_list) / len(psnr_list)
    print(train_psnr)
    # --- Save the network parameters --- #
    torch.save(net.state_dict(), 'haze_add'.format())
    torch.save(net_data.state_dict(), 'haze_data'.format())
    print(3)
    # --- Use the evaluation model in testing --- #

    val_psnr, val_ssim = validation(net, val_data_loader, device)
    test_psnr, test_ssim = validation(net, test_data_loader, device)
    print(val_psnr)
    print(old_val_psnr)
    # --- update the network weight --- #
    if test_psnr >= old_test_psnr:
        torch.save(net.state_dict(), 'haze_add_best_test_{}'.format(epoch))
        torch.save(net_data.state_dict(), 'haze_data_best_test_{}'.format(epoch))
        old_test_psnr = test_psnr
    if val_psnr >= old_val_psnr:
        torch.save(net.state_dict(), 'haze_add_best_{}'.format(epoch))
        torch.save(net_data.state_dict(), 'haze_data_best_{}'.format(epoch))
        old_val_psnr = val_psnr

    print(epoch)
    torch.save(baseline,'baseline'.format())
    torch.save(epoch, 'epoch'.format())
    with open('val_psnr.txt','a+') as f:
        f.write(str(val_psnr) + '\n')
    with open('val_ssim.txt', 'a+') as f:
        f.write(str(val_ssim) + '\n')
    with open('test_psnr.txt', 'a+') as f:
        f.write(str(test_psnr) + '\n')
    with open('test_ssim.txt', 'a+') as f:
        f.write(str(test_ssim) + '\n')
