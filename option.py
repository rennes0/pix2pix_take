import argparse
import torch
import torchvision.transforms as transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DISC = "disc.pth"
CHECKPOINT_GEN = "gen.pth"
def parse_args():
    desc = "Pytorch implementation of GAN collections"
    parser = argparse.ArgumentParser(description=desc)

    #config
    parser.add_argument('--split', type=str, default='', help='The split flag for svhn and stl10')
    parser.add_argument('--num_epoch', type=int, default=10, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=10, help='The size of batch')
    parser.add_argument('--input_size', type=int, default=256, help='The size of input image')
    parser.add_argument('--dataroot', default='./dataset/Plane_data_750x800', help='the path of the dataset')
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=[64, 128, 256, 512], help='# of discrim filters in the first conv layer')
    parser.add_argument('--lrG', type=float, default=0.0002)
    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--L1_LAMBDA', type=int, default=100)
    parser.add_argument('--in_and_out_change', type=str, default='BtoA')
    parser.add_argument('--gan_type' , type = str, default='pix2pix' ,choices=['wgangp'], help= 'Compare different model')
    parser.add_argument('--exp', type=str, default='C307', choices=['C315','C330','C615','C1015'], help='Name of experiment')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--benchmark_mode', type=bool, default=True)
    parser.add_argument('--use_gp',action='store_false',help='not input--> True, input <--use_gp> --> False')
    # label encoding
    parser.add_argument('--discrete_column', nargs="*", type=str, default=["AR", "HR", "VR", "DI"], help='The discrete label of input')
    parser.add_argument('--continuous_column', nargs="*", type=str, default=[], help='The continuous label of input')
    parser.add_argument('--add_attribute', type=str, default='d_DI' , choices=['d_DI_AR' ,'d_DI_HR' ,'d_DI_VR'] , help='The discrete label')
    parser.add_argument('--lambda_gp', type=int, default=10)
    return parser.parse_args()  #check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):
    discrete = ""
    for i, dc in enumerate(args.discrete_column):
        if i == 0:
            discrete += dc
        else:
            discrete += f"_{dc}"
    

    continuous = ""
    for i, cc in enumerate(args.continuous_column):
        if i == 0:
            continuous += cc
        else:
            continuous += f"_{cc}"
    #args.exp = f"s{args.input_size}x{args.input_size}_b{args.batch_size}_e{args.num_epoch}_lrG{args.lrG}_lrD{args.lrD}_d{discrete}_c{continuous}"
    
    # --result_dir
    # args.result_dir = os.path.join(args.result_dir, args.dataset, args.gan_type, args.exp)
    # if not os.path.exists(os.path.join(args.result_dir, "progress")):
    #     os.makedirs(os.path.join(args.result_dir, "progress"))

    # if not os.path.exists(os.path.join(args.result_dir, "model")):
    #     os.makedirs(os.path.join(args.result_dir, "model"))

    # --epoch
    try:
        assert args.num_epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

opt = parse_args()
TRANSFORM = transforms.Compose([transforms.Resize((opt.input_size, opt.input_size)), 
                                transforms.ToTensor(), 
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                    ])