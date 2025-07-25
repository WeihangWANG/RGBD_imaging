import argparse, os, glob
from cmath import log10
from statistics import mode
from tkinter import N
from cv2 import HoughLines
from matplotlib.pyplot import axis
import torch,pdb
import math, random, time
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from srresnet_unet3 import _NetG,_NetD,_NetDedge
from dataset_dep import DatasetFromHdf5
from torchvision.utils import save_image
import torch.utils.model_zoo as model_zoo
from random import randint, seed
import random
import cv2
# from DeepToF_model import DeepToF

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet") 
parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=200, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=100, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to resume model (default: none")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, (default: 1)")
parser.add_argument("--pretrained", default="", type=str, help="Path to pretrained model (default: none)")
parser.add_argument("--vgg_loss", action="store_true", help="Use content loss?")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--trainset", default="../../tofdata/h5data/zju_realworld_qua/", type=str, help="dataset name")
parser.add_argument("--sigma", default=5, type=int)
parser.add_argument("--beta", default=[1,10,20,30,50,60,70,80,90], type=list)
parser.add_argument("--alpha", default=[40], type=list)
parser.add_argument("--noise_sigma", default=10, type=int, help="standard deviation of the Gaussian noise (default: 10)")

# device_ids = [0]
C1 = 299792458.0 / (4 * np.pi * 75000000.0)
C2 = 299792458.0 / (4 * np.pi * 100000000.0)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num

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

def test(model):
    root_dir = '/home/wangjun/2021zju/ToFDatasets/RealWorld/test/'

    data_files = sorted([file for file in os.listdir(root_dir) if file.endswith('.npy')])
    num_images = len(data_files)

    p=0
    ps = 0


    with torch.no_grad():
        for index in range(0, len(data_files)):
            # sys.stdout.write('\rProcessed :: {} out of :: {}'.format(index + 1, len(data_files)))
            # sys.stdout.flush()

            data = np.load(root_dir + data_files[index], allow_pickle=True).item()
            corr = data['corr']  # CHW, the 4 corr
            gt = data['gt']  # HW
            amp = data['amp']  # HW
            depth_tra = data['depth_tra']
            # rgb = imageio.imread(root_dir + data_files[index].split('.')[0] + '.png')  # HWC
            
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
            # rgb = np.array(rgb).astype(dtype=np.float32)
            gt = np.array(gt).astype(dtype=np.float32)
            amp = np.array(amp).astype(dtype=np.float32)
            depth = np.array(depth_tra).astype(dtype=np.float32)

            # rgb = rgb / 255.0

            mask = np.where(gt != 0, 1, 0)

            gt = np.expand_dims(gt, 0)
            corr_ = np.expand_dims(corr_, 0)
            amp = np.expand_dims(amp, 0)
            depth = np.expand_dims(depth, 0)

            im_input = torch.cat((torch.from_numpy(corr_).float(), torch.from_numpy(depth).float(), torch.from_numpy(amp).float()), 1)
            im_gt = torch.from_numpy(gt).float()

            im_gt = im_gt.cuda()
            im_input = im_input.cuda()

            im_output = model(im_input)

            im_output = (im_output).cpu().numpy()
            im_output = torch.from_numpy(im_output).float()
            noisy_input = torch.from_numpy(depth).float()
            gt_full = torch.from_numpy(gt).float()

            noisy_input = np.squeeze(noisy_input)
            im_output = im_output.squeeze()
            gt_full = np.squeeze(gt_full)

            pp=MAE(im_output,gt_full, mask)
            pps=PSNR(im_output,gt_full, mask)

            p+=pp
            ps+=pps
    return p/num_images, ps/num_images

def main():
    global opt, model, netContent
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda: 
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        # ids = [0, 1, 2]
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    
    
    for beta in opt.beta:
        for alpha in opt.alpha:
            file = open('./checksample/log_r.txt','a')
            file.write(str(beta)+'_'+str(alpha)+':\n')
            mae = 10.0
            ps = 0
            opt.seed = random.randint(1, 10000)
            print("Random Seed: ", opt.seed)
            torch.manual_seed(opt.seed)
            if cuda:
                torch.cuda.manual_seed(opt.seed)

            cudnn.benchmark = True

            print("===> Loading datasets")
            data_list = glob.glob(opt.trainset+"*.h5")
            print(data_list)

            print("===> Building model")
            model = _NetG()
            # model = DeepToF()
            discr = _NetD()
            # discr_2 = _NetDedge()
            criterion = nn.MSELoss(size_average=True)
            #网络参数数量
            # a,b=get_parameter_number(model)
            # print(model)
            # print(a,b)
            print("===> Setting GPU")
            if cuda:
                model = model.cuda()
                discr = discr.cuda()
                criterion = criterion.cuda()

                
                # model = torch.nn.DataParallel(model, device_ids=device_ids)
                # # 模型加载到设备0
                # model = model.cuda(device=device_ids[0])
                # discr = torch.nn.DataParallel(discr, device_ids=device_ids)
                # discr = discr.cuda(device=device_ids[0])
                # criterion = torch.nn.DataParallel(criterion, device_ids=device_ids)
                # criterion = criterion.cuda(device=device_ids[0])

                # discr_2 = discr_2.cuda()

            # optionally resume from a checkpoint
            if opt.resume:
                if os.path.isfile(opt.resume):
                    print("=> loading checkpoint '{}'".format(opt.resume))
                    checkpoint = torch.load(opt.resume)
                    opt.start_epoch = checkpoint["epoch"] + 1
                    model.load_state_dict(checkpoint["model"].state_dict())
                    discr.load_state_dict(checkpoint["discr"].state_dict())
                    # discr_2.load_state_dict(checkpoint["discr_2"].state_dict())
                else:
                    print("=> no checkpoint found at '{}'".format(opt.resume))

            # optionally copy weights from a checkpoint
            if opt.pretrained:
                if os.path.isfile(opt.pretrained):
                    print("=> loading model '{}'".format(opt.pretrained))
                    weights = torch.load(opt.pretrained)
                    model.load_state_dict(weights['model'].state_dict())
                    discr.load_state_dict(weights['discr'].state_dict())
                    # discr_2.load_state_dict(checkpoint["discr_2"].state_dict())
                else:
                    print("=> no model found at '{}'".format(opt.pretrained))

            print("===> Setting Optimizer")
            G_optimizer = optim.RMSprop(model.parameters(), lr=opt.lr/2)
            D_optimizer = optim.RMSprop(discr.parameters(), lr=opt.lr)
            # D_optimizer2 = optim.RMSprop(discr_2.parameters(), lr=opt.lr)

            print("===> Training")
            for epoch in range(opt.start_epoch, opt.nEpochs + 1):
                for data_name in data_list:
                    train_set = DatasetFromHdf5(data_name)
                    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, \
                        batch_size=opt.batchSize, shuffle=True)
                    a,b,c,d=train(training_data_loader, G_optimizer, D_optimizer, model, discr, criterion, epoch, alpha, beta)
                mae, ps = save_checkpoint(model, discr, epoch, mae, ps, alpha, beta)
                if epoch == opt.nEpochs:
                    file.write(str(mae)+'_'+str(ps)+'\n')
            file.close()

   
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr 

def grad_x(img):
    # img=torch.squeeze(img)
    padding = (
        0,1,
        0,0
    )
    img1 = F.pad(img, padding, mode='replicate')
    grad_x = img1[:,:,:,:-1]-img1[:,:,:,1:]
    return grad_x

def grad_y(img):
    # img=torch.squeeze(img)
    padding = (
        0,0,
        0,1
    )
    img2 = F.pad(img, padding, mode='replicate')
    grad_y = img2[:,:,:-1,:]-img2[:,:,1:,:]
    return grad_y


def smooth_loss(depth, amp, mask):
    # print(depth.shape, amp.shape)
    disp_x_grad = grad_x(depth)
    disp_y_grad = grad_y(depth)

    image_x_grad = grad_x(amp)
    image_y_grad = grad_y(amp)

    #e^(-|x|) weights, gradient negatively exponential to weights
    #average over all pixels in C dimension
    #but supposed to be locally smooth?
    weights_x = torch.exp(-torch.abs(image_x_grad))
    weights_y = torch.exp(-torch.abs(image_y_grad))

    smoothness_x = disp_x_grad * weights_x
    smoothness_y = disp_y_grad * weights_y

    # print(smoothness_x.shape, smoothness_y.shape)

    smoothness = torch.abs(smoothness_x) + torch.abs(smoothness_y)
    # print(smoothness.shape)

    smoothness_loss = torch.mean(smoothness[mask > 0])
    # print("smoothness_loss: ", smoothness_loss)

    return smoothness_loss

def pre_process(hole_sigma=0.1, th = 0.4):
    height = 120
    width = 160
    img = np.zeros((height, width), np.uint8)
    
    # Set size scale
    size = int((width + height) * hole_sigma)
    if width < 30 or height < 40:
        raise Exception("Width and Height of mask must be at least 64!")
        
    # Draw random circles
    for _ in range(randint(1, 5)):
        x1, y1 = randint(1, width), randint(1, height)
        radius = randint(5, size)
        cv2.circle(img,(x1,y1),radius,(1), -1)
            
    # Draw random ellipses
    for _ in range(randint(1, 5)):
        x1, y1 = randint(1, width), randint(1, height)
        s1, s2 = randint(1, width), randint(1, height)
        a1, a2, a3 = randint(5, 180), randint(5, 180), randint(5, 180)
        thickness = randint(5, size)
        cv2.ellipse(img, (x1,y1), (s1,s2), a1, a2, a3,(1), thickness)

    noise = np.random.normal(size=img.shape)
    mask2 = np.where(noise < th, 1, 0)
    mask = img * mask2
    mask = 1 - mask
    mask = np.expand_dims(mask, axis=(0, 1))
    return mask

def train(training_data_loader, G_optimizer, D_optimizer, model, discr, criterion, epoch, alpha_, beta):

    lr = adjust_learning_rate(D_optimizer, epoch-1)
    mse = []
    Gloss=[]
    Dloss = []
    l1=[]
    c_loss= []
    s_loss= []
    for param_group in G_optimizer.param_groups:
        param_group["lr"] = lr/2
    for param_group in D_optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, D_optimizer.param_groups[0]["lr"]))

    for iteration, batch in enumerate(training_data_loader, 1):

        target_ = Variable(batch[2])
        rgb_input = Variable(batch[0])
        d_input = Variable(batch[1])
        corr_input = Variable(batch[4])
        amp_input = Variable(batch[3])

        # amp_input = np.where(amp_input > 2.0, 0, amp_input)
        # amp_input = torch.from_numpy(amp_input).float()

        # rng_stddev = np.random.uniform(0.01, 60.0/255.0,[1,1,1])
        # noise = np.random.normal(size=target_.shape) * rng_stddev
        # # noise = np.random.normal(size=target_.shape) * opt.noise_sigma/255.0   
        # noise = torch.from_numpy(noise).float()

        hole_mask = pre_process()
        hole_mask = torch.from_numpy(hole_mask).float()

        for k in range(len(corr_input) - 1):
            hole_mask_ = pre_process()
            hole_mask_ = torch.from_numpy(hole_mask_).float()
            hole_mask = torch.cat((hole_mask, hole_mask_), 0)

        if opt.cuda:
            target_ = (target_/10.0).cuda()
            rgb_input = rgb_input.cuda()
            d_input = (d_input/10.0).cuda()
            corr_input = corr_input.cuda()
            amp_input = amp_input.cuda()
            hole_mask = hole_mask.cuda()

        # print(len(corr_input))
        mask_loss = torch.where(d_input > 0, 1, 0)
        corr_mask = torch.where(corr_input > 0, 1, 0)

        # corr_input = torch.mul(corr_input, hole_mask)
        # d_input = torch.mul(d_input, hole_mask)
        # amp_input = torch.mul(amp_input, hole_mask)
        corr_input = torch.multiply(corr_input, hole_mask)
        d_input = torch.multiply(d_input, hole_mask)

        input = torch.cat((corr_input, d_input, amp_input), 1)
        target = torch.cat((rgb_input, target_), 1)

        # train discriminator D
        discr.zero_grad()

        D_result = discr(target).squeeze()
        D_real_loss = -D_result.mean()

        G_result = model(input)
        # print("===============", G_result.shape)
        G_result_ = torch.cat((rgb_input, G_result), 1)
        D_result = discr(G_result_.data).squeeze()

        D_fake_loss = D_result.mean()


        D_train_loss = D_real_loss + D_fake_loss
        Dloss.append(D_train_loss.data)

        D_train_loss.backward()
        D_optimizer.step()


        #gradient penalty
        discr.zero_grad()
        alpha = torch.rand(target.size(0), 1, 1, 1)
        alpha1 = alpha.cuda().expand_as(target)
        interpolated1 = Variable(alpha1 * target.data + (1 - alpha1) * G_result.data, requires_grad=True)
        
        out = discr(interpolated1).squeeze()

        grad = torch.autograd.grad(outputs=out,
                                   inputs=interpolated1,
                                   grad_outputs=torch.ones(out.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

        # Backward + Optimize
        gp_loss = 10 * d_loss_gp

        gp_loss.backward()
        D_optimizer.step()



        # train generator G
        discr.zero_grad()
        model.zero_grad()

        G_result = model(input)
        G_result_ = torch.cat((rgb_input, G_result), 1)
        D_result = discr(G_result_).squeeze()

        # amp_mask = torch.where(amp_input > 0.0, 1, 0)
        
        mse_loss = (torch.mean(((G_result- d_input)[mask_loss > 0])**2))**0.5
        mse.append(mse_loss.data)

        l1_mask = mask_loss
        l1_loss = torch.mean(abs(G_result- d_input)[l1_mask > 0])
        l1.append(l1_loss)

        ## corr_loss
        # corr_mask = torch.mul(corr_mask, amp_mask)
        pd1 = G_result * 10.0 / C1 ## 75M corr_1,2
        pd2 = G_result * 10.0 / C2 ## 100M corr_3,4
        sin1 = torch.sin(pd1) 
        cos1 = torch.cos(pd1)
        sin2 = torch.sin(pd2) 
        cos2 = torch.cos(pd2)
        corr_out = torch.cat((sin1,cos1,sin2,cos2),1) * amp_input
        corr_loss = torch.mean(abs(corr_out- corr_input)[corr_mask > 0])
        c_loss.append(corr_loss)

        disp_loss = smooth_loss(G_result, amp_input, mask_loss)
        s_loss.append(disp_loss)

        # print(D_result.mean(), D_result2.mean())
        G_train_loss = - D_result.mean() + opt.sigma / 100.0 * corr_loss + beta * l1_loss + alpha_ * disp_loss
        Gloss.append(G_train_loss)
        G_train_loss.backward()
        G_optimizer.step()
        
        if iteration % 30 == 0:
            print("===> Epoch[{}]({}/{}): Loss_G: {:.5}, Loss_corr: {:.5}, Loss_l1: {:.5}, Loss_s: {:.5}".format(epoch, iteration, len(training_data_loader), G_train_loss.data, corr_loss.data, l1_loss.data, disp_loss.data))
    save_image(G_result.data, './checksample/output.png')
    save_image(d_input.data, './checksample/input.png')
    save_image(target_.data, './checksample/gt.png')
    save_image(rgb_input.data, './checksample/rgb.png')

    return torch.mean(torch.FloatTensor(c_loss)),torch.mean(torch.FloatTensor(Gloss)),torch.mean(torch.FloatTensor(l1)),torch.mean(torch.FloatTensor(s_loss))
   
def save_checkpoint(model, discr, epoch, mae, ps, alpha, beta):
    if epoch%10==0:
        err, p=test(model)
        if err < mae:
            model_out_path = "checkpoint/" + "model_denoise_r_"+str(opt.nEpochs)+"_"+str(opt.beta)+"_"+str(opt.alpha)+".pth"
            state = {"epoch": epoch ,"model": model, "discr": discr}
            if not os.path.exists("checkpoint/"):
                os.makedirs("checkpoint/")

            mae = err
            ps = p

            torch.save(state, model_out_path)

            print("Checkpoint saved to {}".format(model_out_path))
    print("alpha:", alpha, "beta:", beta)
    print("MAE:", mae, "PSNR:", ps)
    return mae, ps

if __name__ == "__main__":
    main()
