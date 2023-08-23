import torch
import option
from PIL import Image
from torchvision.utils import save_image
import os
from model import Generator
from utils import load_checkpoint, save_some_examples
import matplotlib.pyplot as plt
def prediction(netG, epoch,exp,choice,
               c_path='./dataset/Plane_data_750x800/train/C/S__22593539_0.jpg',
               a_path='./dataset/Plane_data_750x800/train/A/C615_DI=0.0NW.JPG',
               opt = option.parse_args(),
               condition_len = 18,
               di = 9):
    dis_label = torch.zeros(1, condition_len).type(torch.int)
    con_label = torch.empty(1, 0)

    if exp == 'C307':
        temp = torch.tensor([0,1,0,0,1,1,0,0])                    
    elif exp == 'C315':
        temp = torch.tensor([0,1,0,1,0,0,1,0])
    elif exp == 'C330':
        temp = torch.tensor([0,1,0,1,0,0,0,1])
    elif exp == 'C615':
        temp = torch.tensor([0,0,1,1,0,0,1,0])
    else: # C1050 x2
        temp = torch.tensor([1,0,0,1,0,0,1,0])
    dis_label[:,:8] = temp
    # Define DI
    dis_label[: , 8+di] = 1
    
    # To device
    dis_label = dis_label.to(option.DEVICE)
    con_label = con_label.to(option.DEVICE)
    # state_dict = torch.load(path)["state_dict"]
    # netG.load_state_dict(state_dict)
    if use_gp==True:
        load_checkpoint(f'./w&b/use_gp:{use_gp}/{epoch}_gen.pth', model=netG)
    else:
        load_checkpoint(f'./w&b/use_gp:{use_gp}/{epoch}_gen.pth', model=netG)
    
    #print(msg)
    img = Image.open(choice)
    img = option.TRANSFORM(img).to(option.DEVICE)
    with torch.no_grad():
        output = netG(img.unsqueeze(0), dis_label, con_label).squeeze()
    output = output * 0.5 + 0.5
    save_image(output.detach().cpu(), f"test_{epoch}.png")
    netG.train()




c_path='./dataset/Plane_data_750x800/train/C/S__22593539_0.jpg'
exp = 'C615'
a_path=f'./dataset/Plane_data_750x800/train/A/{exp}_DI=0.0SE.JPG'
opt = option.parse_args()
condition_len = 18

di = 0
dataroot = opt.dataroot
use_gp = True
epoch = '1000'
choice= c_path
netG = Generator(input_nc=opt.input_nc + condition_len, 
                        ngf=opt.ngf,
                        device= option.DEVICE).to(option.DEVICE)
prediction(netG,epoch,exp,choice,
               c_path,
               a_path,
               opt,
               condition_len,
               di)

#save_some_examples(netG, dataroot, exp,use_gp,epoch)
# use_gp =True
# c_path='./dataset/Plane_data_750x800/train/C/S__22593539_0.jpg'
# path = "./w&b/1000_gen.pth"
# opt = option.parse_args()
# condition_len = 18
# exp = 615
# di = 9
# dis_label = torch.zeros(1, condition_len)
# con_label = torch.empty(1, 0)

# if exp == 'C307':
#     dis_label[:, :8] = torch.tensor([0,1,0,0,1,1,0,0])                    
# elif exp == 'C315':
#     dis_label[:, :8] = torch.tensor([0,1,0,1,0,0,1,0])
# elif exp == 'C330':
#     dis_label[:, :8] = torch.tensor([0,1,0,1,0,0,0,1])
# elif exp == 'C615':
#     dis_label[:, :8] = torch.tensor([0,0,1,1,0,0,1,0])
# else: # C1050 x2
#     dis_label[:, :8] = torch.tensor([1,0,0,1,0,0,1,0])

# # Define DI
# dis_label[:7+di] = 1

# # To device
# dis_label = dis_label.to(option.DEVICE)
# con_label = con_label.to(option.DEVICE)

# netG = Generator(input_nc= opt.input_nc + condition_len , 
#                     ngf=opt.ngf,
#                     device= option.DEVICE).to(option.DEVICE)
# state_dict = torch.load(path)["state_dict"]
# netG.load_state_dict(state_dict)
# netG.eval()
# #print(msg)
# img = Image.open(c_path)
# img = option.TRANSFORM(img).to(option.DEVICE)
# with torch.no_grad():
#     output = netG(img.unsqueeze(0), dis_label, con_label).squeeze()
# output = output * 0.5 + 0.5
# save_image(output.detach().cpu(), f"test_{use_gp}.png")
# netG.train()