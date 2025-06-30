import argparse
import os, pdb
import torch, cv2
from torch._C import device
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import time, math, glob
import scipy.io as scio
from PIL import Image
# from ssim import calculate_ssim_floder
from torchvision.utils import save_image, make_grid
import imageio

parser = argparse.ArgumentParser(description="PyTorch SRResNet Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="./checkpoint/model_denoise_200_30.pth", type=str, help="model path")
parser.add_argument("--dataset", default="./KODAK", type=str, help="dataset name, Default: KODAK")
parser.add_argument("--save", default="./results", type=str, help="savepath, Default: results")
parser.add_argument("--noise_sigma", default=50, type=int, help="standard deviation of the Gaussian noise, Default: 25")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids")

def plane_correction(fov, img_size, fov_flag=True):
    x, y = np.meshgrid(np.linspace(0, img_size[1] - 1, img_size[1]),
                       np.linspace(0, img_size[0] - 1, img_size[0]))
    if fov_flag:
        fov_pi = 63.5 * np.pi / 180
        flen = (img_size[1] / 2.0) / np.tan(fov_pi / 2.0)
    else:
        flen = fov

    x = (x - img_size[1] / 2.) / flen
    y = (y - img_size[0] / 2.) / flen
    norm = 1. / np.sqrt(x ** 2 + y ** 2 + 1.)

    return norm

def PSNR(pred, gt, msk=None, shave_border=0):
    if msk is None:
        # depth_kinect_msk = np.where(gt < 1.0, 1, 0)
        # depth_kinect_msk_tmp = np.where(gt > 10.0 / 4095.0, 1, 0)
        # msk = depth_kinect_msk * depth_kinect_msk_tmp
        msk = np.ones_like(pred)
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    imdff = imdff[msk > 0]
    rmse = math.sqrt((imdff ** 2).mean())
    if rmse == 0:
        return 100  
    return 20 * math.log10(1.0 / rmse)

def MAE(pred, gt, msk=None, shave_border=0):
    if msk is None:
        # depth_kinect_msk = np.where(gt < 1.0, 1, 0)
        # depth_kinect_msk_tmp = np.where(gt > 10.0 / 4095.0, 1, 0)
        # msk = depth_kinect_msk * depth_kinect_msk_tmp
        msk = np.ones_like(pred)
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    imdff = imdff[msk > 0]
    mae_ = abs(imdff).mean()
    return mae_



opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpus)
cuda = True#opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

if not os.path.exists(opt.save):
    os.mkdir(opt.save)

model = torch.load(opt.model)["model"]

root_dir = '/home/wangjun/2021zju/ToFDatasets/RealWorld/test/'

data_files = sorted([file for file in os.listdir(root_dir) if file.endswith('.npy')])
num_images = len(data_files)

p=0
p2=0
image_size = (480, 640)

ps = 0
ps2 = 0


with torch.no_grad():
    for index in range(0, len(data_files)):
        # sys.stdout.write('\rProcessed :: {} out of :: {}'.format(index + 1, len(data_files)))
        # sys.stdout.flush()

        data = np.load(root_dir + data_files[index], allow_pickle=True).item()
        corr = data['corr']  # CHW, the 4 corr
        gt = data['gt']  # HW
        amp = data['amp']  # HW
        depth_tra = data['depth_tra']
        rgb = imageio.imread(root_dir + data_files[index].split('.')[0] + '.png')  # HWC

        # corr = np.transpose(corr, (1, 2, 0))  # HWC
        # corr /= amp[:, :, np.newaxis]
        # corr_ = np.zeros([4, 480, 640])
        # for i in range(4):
        #     corr_[i, :, :] = corr[:, :, i]
        
        H, W = gt.shape
        # print(H, W)
        corr = np.transpose(corr, (1, 2, 0))  # HWC
        # corr /= amp[:, :, np.newaxis]

        input_data = np.zeros([H, W, 6], dtype=np.float32)
        input_data[:, :, 0:4] = corr
        input_data[:, :, 4] = amp
        input_data[:, :, 5] = depth_tra
        nan_and_inf = np.isnan(input_data) | np.isinf(input_data)
        input_data[nan_and_inf] = 0

        corr = input_data[:, :, 0:4]
        amp = input_data[:, :, 4]
        depth_tra = input_data[:, :, 5]

        corr_ = np.zeros([4, H, W])
        for i in range(4):
            corr_[i, :, :] = corr[:, :, i]
        amp = amp.reshape([1, 480, 640])
        depth_tra = (depth_tra/10.0).reshape([1, H, W])

        gt = gt.reshape([1, H, W])
        gt = (gt/10.0).astype(np.float32)

        corr_ = np.array(corr_).astype(dtype=np.float32)
        rgb = np.array(rgb).astype(dtype=np.float32)
        gt = np.array(gt).astype(dtype=np.float32)
        amp = np.array(amp).astype(dtype=np.float32)
        depth = np.array(depth_tra).astype(dtype=np.float32)

        # depth = np.where(amp < 0.002, 0, depth)

        rgb = rgb / 255.0

        mask = np.where(gt != 0, 1, 0)

        # corr = corr.reshape([4, 480, 640])
        # gt = gt.reshape([1, 480, 640])
        # amp = amp.reshape([1, 480, 640])
        # depth = depth.reshape([1, 480, 640])
        # mask = mask.reshape([1, 480, 640])
        # rgb_ = np.zeros([3, 480, 640])
        # for z in range(3):
        #     rgb_[z, :, :] = rgb[:, :, z]
        
        # output_full = noisy.copy()
        # rgb_ = np.expand_dims(rgb_, 0)
        gt = np.expand_dims(gt, 0)
        corr_ = np.expand_dims(corr_, 0)
        amp = np.expand_dims(amp, 0)
        depth = np.expand_dims(depth, 0)

        im_input = torch.cat((torch.from_numpy(corr_).float(), torch.from_numpy(depth).float(), torch.from_numpy(amp).float()), 1)
        # im_input = torch.from_numpy(noisy).float()
        im_gt = torch.from_numpy(gt).float()

        
        if cuda:
            model = model.cuda()
            im_gt = im_gt.cuda()
            im_input = im_input.cuda()
        else:
            model = model.cpu()
        


        im_output = model(im_input)

        # output_full = torch.from_numpy(output_full*2.0).float()
        im_output = (im_output).cpu().numpy()
        im_output = torch.from_numpy(im_output).float()
        noisy_input = torch.from_numpy(depth).float()
        gt_full = torch.from_numpy(gt).float()
        # print('==============================:',im_output.shape)

        noisy_input = np.squeeze(noisy_input)
        # mask_ = np.squeeze(mask_)
        im_output = im_output.squeeze()
        gt_full = np.squeeze(gt_full)

        pp=MAE(im_output,gt_full, mask)
        pp2=MAE(noisy_input,gt_full, mask)
        pps=PSNR(im_output,gt_full, mask)
        pps2=PSNR(noisy_input,gt_full, mask)

        # if pp > 0.1:
        #     pp = 0
        #     pps = 0
        #     num_images -= 1

        p+=pp
        p2+=pp2
        ps+=pps
        ps2+=pps2
        print(pp, pp2, pps, pps2)
        # save_image(im_output.data, opt.save+'/'+'%03d.png'%i)

        im_output = im_output.squeeze()
        im_output = im_output.type(torch.float32).numpy()
        im_output.tofile(opt.save+'/'+'%03d.png'%index)

print("Average MAE:",p/num_images)
print("Average input MAE:",p2/num_images)
print("Average PSNR:",ps/num_images)
print("Average input PSNR:",ps2/num_images)
