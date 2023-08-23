import torch
import option
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
from PIL import Image
from dataloader import DamageIndexDataset
from time import time
import pandas as pd
import csv
from fid_score import calculate_fid_given_paths

#     AR_10   AR_3   AR_6  HR_1.106  HR_1.282  VR_0.742  VR_1.444  VR_2.889  DI_0.1  DI_0.2  DI_0.3  DI_0.4  DI_0.5  DI_0.6  DI_0.7  DI_0.8  DI_0.9  DI_1.0
# 0    True  False  False      True     False     False      True     False    True   False   False   False   False   False   False   False   False   False
# 1    True  False  False      True     False     False      True     False    True   False   False   False   False   False   False   False   False   False
# 2    True  False  False      True     False     False      True     False   False    True   False   False   False   False   False   False   False   False
# 3    True  False  False      True     False     False      True     False   False    True   False   False   False   False   False   False   False   False
# 4    True  False  False      True     False     False      True     False   False   False    True   False   False   False   False   False   False   False
# ..    ...    ...    ...       ...       ...       ...       ...       ...     ...     ...     ...     ...     ...     ...     ...     ...     ...     ...
# 95  False  False   True      True     False     False      True     False   False   False   False   False   False   False   False    True   False   False
# 96  False  False   True      True     False     False      True     False   False   False   False   False   False   False   False   False    True   False
# 97  False  False   True      True     False     False      True     False   False   False   False   False   False   False   False   False    True   False
# 98  False  False   True      True     False     False      True     False   False   False   False   False   False   False   False   False   False    True
# 99  False  False   True      True     False     False      True     False   False   False   False   False   False   False   False   False   False    True
# [100 rows x 18 columns]

def save_some_examples(netG, dataroot, exp,use_gp, epoch):
    GP = f'use_gp:{use_gp}'
    os.makedirs(GP, exist_ok= True)
    folder = f'./{GP}/gen_{exp}'
    os.makedirs(folder, exist_ok= True)
    os.makedirs(os.path.join(folder,'input'), exist_ok= True)
    os.makedirs(os.path.join(folder,'fake'), exist_ok= True)
    path = os.path.join(dataroot , 'train' )
    file_list= os.listdir(os.path.join(path, "B"))
    for name in file_list:
        if name.split('_')[0] != exp:
            continue
        label = exp  # Extract the label (C307, C118)
        direction = name.split("=")[1][3:5]  # Extract the direction (NW, SE)
        nameA = os.path.join(path, 'A', f"{label}_DI=0.0{direction}.JPG")
        # nameB = os.path.join(path, 'B', name)
        # imgB = option.TRANSFORM(Image.open(nameB))
        imgA = option.TRANSFORM(Image.open(nameA))
        with torch.no_grad():
            dis_label = torch.zeros(10,18).type(torch.int)
            #     AR_10   AR_3   AR_6  HR_1.106  HR_1.282  VR_0.742  VR_1.444  VR_2.889  DI_0.1  DI_0.2  DI_0.3  DI_0.4  DI_0.5  DI_0.6  DI_0.7  DI_0.8  DI_0.9  DI_1.0
            # if add_attribute == 'd_DI':
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
            for j in range(10):
                        dis_label[j , :8]= temp
                        dis_label[j , 8+j]= 1
            # elif add_attribute =='d_DI_AR':
            #     dis_label = torch.tensor([[1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]])
            # elif add_attribute =='d_DI_HR':
            #     dis_label = torch.tensor([[0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1]])
            # elif add_attribute =='d_DI_VR':
            #     dis_label = torch.tensor([[0,0,0,0,0,1,1,0,1,1,1,1,1,1,1,1,1,1]])            
            con_label = torch.empty((dis_label.shape[0], 0))
            #loadcheckpoint
            if use_gp==True:
                load_checkpoint(f'./w&b/use_gp:{use_gp}/{epoch}_gen.pth', model=netG)
            else:
                load_checkpoint(f'./w&b/use_gp:{use_gp}/{epoch}_gen.pth', model=netG)
            batch_size , _ =dis_label.shape
            nc, input_size, _ = imgA.shape
            imgA = imgA.unsqueeze(0).expand(batch_size, nc, input_size, input_size)
            imgA = imgA.to(option.DEVICE)
        #     dis_label = numpy.array([False,  True, False,  True, False, False, False,  True, False, False,
        #  False, False, False, False, False, False,  True, False])
        #     con_label = numpy.empty((dis_label.shape[0],0))
            y_fake = netG(imgA, dis_label, con_label)
            y_fake = y_fake * 0.5 + 0.5  # remove normalization#
            save_image(y_fake.detach().cpu(), f"y_gen{epoch}.png", nrow = 10) #os.path.join(folder ,'fake', f"y_gen.png")
            #save_image(imgA * 0.5 + 0.5 , os.path.join(folder , 'input', f"input_{epoch}.png"), nrow = 10)
        netG.train()


def save_checkpoint(model, optimizer, filename):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer=None, lr=0.01):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=option.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer != None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    model.eval()
    return model
    

def save_loss(G_list,D_list,exp,use_gp):
    folder = f'./use_gp:{use_gp}/gen_{exp}'
    fid = calculate_fid_given_paths(real_path=os.path.join(folder,'input'),
                                    fake_path=os.path.join(folder,'fake'),
                                    batch_size=1,
                                    device=option.DEVICE,dims=2048,
                                    num_workers = min(os.cpu_count() , 8))
    fig, ax1 = plt.subplots(figsize=(9,5))
    fig.set_facecolor('white')
    plt.xlabel("iterations", fontsize = 16)
    ax2 = ax1.twinx()
       
    # set G_loss
    ax1.set_ylabel("Loss", fontsize = 16)
    ax1.plot(G_list, color = 'C1',label = 'G')
    ax1.tick_params(axis='y', labelcolor = 'C1')
    #set D_loss
    filter_data = [v for v in D_list if v<=2.0]
    ax2.plot(filter_data, color = 'C0',label = 'D')
    ax2.tick_params(axis='y', labelcolor = 'C0')
    plt.tight_layout()
    if use_gp:
        plt.title(f"Training with GP            FID Score = {fid:.2f}", fontsize = 20)
    else:
        plt.title(f"Training without GP            FID Score = {fid:.2f}", fontsize = 20)
    ax1.set_yticks([0.0,0.5,1.0,1.5,2.0])
    ax2.set_yticks([0.0,0.5,1.0,1.5,2.0])
    # create labels
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2, loc='upper right')
    plt.savefig(f'{folder}/loss{exp}_{use_gp}.svg')

def read_loss(G_list, exp,use_gp):
    folder = f'./use_gp:{use_gp}/gen_{exp}'
    path = os.path.join(folder, f'loss_{exp}_{use_gp}.csv')
    # fid = calculate_fid_given_paths(real_path=os.path.join(folder,'input'),
    #                                 fake_path=os.path.join(folder,'fake'),
    #                                 batch_size=1,
    #                                 device=option.DEVICE,
    #                                 dims=2048,
    #                                 num_workers = min(os.cpu_count() , 8))

    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # writer.writerow(['FID Scores:', fid])
        #writer.writerow(["G_loss", "D_loss", f'use_gp:{use_gp}'])
        # for (g, d, l) in zip(G_list, D_list, loss_list):
        #     writer.writerow([g, d, l])
        writer.writerow(["G_loss"])
        for d in G_list:
            writer.writerow([d])

def gradient_penalty(Discriminator, real, fake, device, real_, dis, con):
    Batch_size, nc, image_size, image_size = real.shape
    alpha = torch.rand((Batch_size, 1, 1, 1)).repeat(1, nc, image_size, image_size).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = Discriminator(interpolated_images, real_, dis, con)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

    
    
# dis_label = torch.tensor([[False,  True, False,  True, False, False, False,  True, False, False,
#          False, False, False, False, False, False,  True, False],
#          [False, False,  True,  True, False, False,  True, False,  True, False,
#          False, False, False, False, False, False, False, False],
#         [False, False,  True,  True, False, False,  True, False, False, False,
#          False, False, False,  True, False, False, False, False],
#         [False, False,  True,  True, False, False,  True, False, False,  True,
#          False, False, False, False, False, False, False, False],
#         [False,  True, False,  True, False, False,  True, False, False, False,
#          False, False,  True, False, False, False, False, False],
#         [False,  True, False,  True, False, False,  True, False, False, False,
#          False, False, False,  True, False, False, False, False],
#         [ True, False, False,  True, False, False,  True, False, False,  True,
#          False, False, False, False, False, False, False, False],
#         [False,  True, False,  True, False, False, False,  True, False,  True,
#          False, False, False, False, False, False, False, False],
#         [False,  True, False,  True, False, False, False,  True, False, False,
#          False, False, False, False,  True, False, False, False],
#         [False, False,  True,  True, False, False,  True, False, False, False,
#          False, False, False, False, False, False,  True, False]])