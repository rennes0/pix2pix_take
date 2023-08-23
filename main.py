import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples, save_loss, read_loss, gradient_penalty
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
#Module from Files 
from model import Generator,Discriminator
from dataloader import DamageIndexDataset
import option
from torchsummary import summary
from predict import prediction
def training(
    netD, netG, optD, optG, l1_loss, bce, g_scaler, d_scaler, dl, G_list, D_list, loss_list, opt, epoch
):
    loop = tqdm(dl, leave=True)

    for idx, (imgA, imgB, dis, con) in enumerate(loop):
        imgA = imgA.to(option.DEVICE)
        imgB = imgB.to(option.DEVICE)
        dis = dis.to(option.DEVICE)
        con = con.to(option.DEVICE)
        
        with torch.cuda.amp.autocast():
            # Train Discriminator
            if opt.use_gp :
                D_real = netD(imgA, imgB, dis, con).reshape(-1)
                imgB_fake = netG(imgA, dis, con)
                D_fake = netD(imgA, imgB_fake.detach(), dis, con).reshape(-1)
                gp = gradient_penalty(Discriminator= netD ,real= imgA, fake= imgB_fake, device= option.DEVICE, real_= imgB, dis= dis, con= con)
                loss = opt.lambda_gp*gp
                D_loss = -torch.mean(D_fake) + torch.mean(D_real) + loss

            else:
                D_real = netD(imgA, imgB, dis, con)
                imgB_fake = netG(imgA, dis, con)
                D_real_loss = bce(D_real, torch.ones_like(D_real))
                D_fake = netD(imgA, imgB_fake.detach(), dis, con)
                D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2

        netD.zero_grad()
        d_scaler.scale(D_loss).backward(retain_graph=True)
        d_scaler.step(optD)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = netD(imgA, imgB_fake, dis, con)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(imgB_fake, imgB) * opt.L1_LAMBDA
            G_loss = G_fake_loss + L1
            loss = L1
        optG.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(optG)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item()
                #D_real=D_real.mean().item(),
                #D_fake=D_fake.mean().item()
            )
            G_list.append(G_loss.item())
            D_list.append(D_loss.item())
            loss_list.append(loss.item())
            #save_img(imgB_fake, opt.saveroot)
    return G_list , D_list, loss_list

def main():
    opt = option.parse_args()
    dataset= DamageIndexDataset(opt=opt,
                                transform = option.TRANSFORM
                                #continuous_column = ["DI"], 
                                #discrete_column=["AR", "HR", "VR"]
                                )
    netD = Discriminator(input_nc= opt.input_nc + opt.output_nc + dataset.label_nc , 
                         ndf=opt.ndf,
                         device= option.DEVICE).to(option.DEVICE)
    # summary(netD, [(3, 256, 256), (3, 256, 256), (18, ), (0, )])
    netG = Generator(input_nc= opt.input_nc + dataset.label_nc , 
                     ngf=opt.ngf,
                     device= option.DEVICE).to(option.DEVICE)
    optD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))
    optG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    G_list = []
    D_list = []
    loss_list = []
    if os.path.isfile(option.CHECKPOINT_DISC) and os.path.isfile(option.CHECKPOINT_GEN) :
        load_checkpoint(option.CHECKPOINT_GEN, netG, optG, lr=opt.lrG)
        load_checkpoint(option.CHECKPOINT_DISC, netD , optD, lr= opt.lrD)
    
    dl = DataLoader(dataset, batch_size = opt.batch_size, shuffle=True)
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(opt.num_epoch+1):
        G_list, D_list, loss_list = training(netD, netG, optD, optG, L1_LOSS, BCE, g_scaler, d_scaler, dl, G_list, D_list, loss_list, opt, epoch)
        if (epoch !=0 and epoch % 100 == 0) or epoch == opt.num_epoch :  
            #torch.save(netG.state_dict(), f'./w&b/{epoch:04d}_{option.CHECKPOINT_GEN}')
            if opt.use_gp:
                os.makedirs(f'./w&b/use_gp:{opt.use_gp}',exist_ok= True)
                save_checkpoint(netG, optG, filename=f'./w&b/use_gp:{opt.use_gp}/{epoch:04d}_{option.CHECKPOINT_GEN}')
                save_checkpoint(netD, optD, filename=f'./w&b/use_gp:{opt.use_gp}/{epoch:04d}_{option.CHECKPOINT_DISC}')
            else:
                os.makedirs(f'./w&b/use_gp:False',exist_ok= True)
                save_checkpoint(netG, optG, filename=f'./w&b/use_gp:False/{epoch:04d}_{option.CHECKPOINT_GEN}')
                save_checkpoint(netD, optD, filename=f'./w&b/use_gp:False/{epoch:04d}_{option.CHECKPOINT_DISC}')
            #save_img(random_fake, opt.saveroot) #np.transpose(imgB_fake.detach().cpu(),(1, 0, 2, 3))
            #save_some_examples(netG , epoch , dataroot=opt.dataroot, add_attribute=opt.add_attribute, exp=opt.exp, use_gp=opt.use_gp)
        #read_loss(G_list ,exp=opt.exp, use_gp=opt.use_gp)
        # if epoch == opt.num_epoch :
        #     prediction(use_gp=opt.use_gp,netG=netG, c_path='./dataset/Plane_data_750x800/train/C/S__22593539_0.jpg',path = "./w&b/1000_gen.pth",opt = option.parse_args(), condition_len = 18, exp = 615, di = 9)
    #save_loss(G_list,D_list ,exp=opt.exp, use_gp=opt.use_gp)
    
    
if __name__ == "__main__":
    main()