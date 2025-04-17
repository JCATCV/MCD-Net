import sys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.mcdnet import InpaintGenerator, Discriminator
import lpips
from config import parser
from core.loss import AdversarialLoss
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn.functional as F
import random
from torch.utils.data import Dataset as dataset

sys.path.append('utils/')
sys.path.append('models/')

style_weights = {
    'conv1_1': 1,
    'conv2_1': 0.8,
    'conv3_1': 0.5,
    'conv4_1': 0.3,
    'conv5_1': 0.1,
}

H, W = 240, 432
def sobel_conv(x):
    # sobel filter
    y = torch.zeros_like(x)
    x = x[:, :, 2:, 2:]
    sobel_x = torch.tensor([[+1, 0, -1], [+2, 0, -2], [+1, 0, -1]], requires_grad=False,dtype = torch.float)
    sobel_y = torch.tensor([[+1, +2, +1], [0, 0, 0], [-1, -2, -1]], requires_grad=False,dtype = torch.float)
    gpu_id = x.get_device()
    sobel_x, sobel_y = sobel_x.to(gpu_id), sobel_y.to(gpu_id)
    sobel_x = sobel_x.view((1,1,3,3))
    sobel_y = sobel_y.view((1,1,3,3))
    # gradients in the x and y direction for both predictions and the target transparencies
    G_x_pred = F.conv2d(x,sobel_x,padding = 0)
    G_y_pred = F.conv2d(x,sobel_y,padding = 0)
    # magnitudes of the gradients
    M_pred = torch.pow(G_x_pred,2)+torch.pow(G_y_pred,2)
    # taking care of nans
    M_pred = (M_pred==0.).float() + M_pred
    y[:,:,2:-2,2:-2] = torch.sqrt(M_pred)
    return y

def cal_edge(x):
    if len(x.shape) == 5:
        x = x.squeeze(0)
    min_val, max_val = torch.min(x.flatten(2), dim=-1)[0].unsqueeze(-1).unsqueeze(-1),  torch.max(x.flatten(2), dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
    x = (x - min_val)/(max_val - min_val)
    return sobel_conv(x)


class train(object):
    def __init__(self):
        self.args = parser.parse_args()
        self.ckpt = {}
        self.model = InpaintGenerator()  
        self.discriminator = Discriminator()
        self.gpu = list(range(torch.cuda.device_count()))
        if True:
            dist.init_process_group(backend='nccl')
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            self.device = torch.device("cuda", local_rank)
            self.model.to(self.device)
            self.discriminator.to(self.device)
            Trainset, Validset = DAVIS(), DAVIS("valid")
            self.discriminator = DDP(self.discriminator, broadcast_buffers = True, find_unused_parameters = False)
            self.train_sampler, self.valid_sampler = DistributedSampler(Trainset), DistributedSampler(Validset)
            self.train_loader = DataLoader(Trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=True,
                                           sampler=self.train_sampler)
            self.valid_loader = DataLoader(Validset, batch_size=args.bacth_size, shuffle=False, num_workers=args.num_workers,pin_memory=True,
                                           sampler=self.valid_sampler)
        
        self.L1 = nn.L1Loss()
        self.L2 = nn.MSELoss()
        self.adversarial_loss = AdversarialLoss(type='hinge')
        self.Lpips_func = lpips.LPIPS(net='vgg')
        self.Lpips_func.to(self.device)
        self.KLLoss = torch.nn.KLDivLoss(reduction="batchmean")

        self.optimG = optim.AdamW(
            params=self.model.parameters(),
            lr=self.args.lr,
            betas=(0,0.999),
            eps=1e-8,
            weight_decay=1e-4,
        )

        self.optimD = optim.Adam(
            params=self.discriminator.parameters(),
            # lr=self.args.lr,
            lr=1e-4,
            betas=(0, 0.99),
        )

        if torch.distributed.get_rank() :
            print('\n*****Start training*****\n')
        for epoch in tqdm(range(1, self.args.epoch + 1)):
            self.train()
            self.adjust_learning_rate(epoch)

        if torch.distributed.get_rank() == 0:
            print('This model will be saved as {}'.format(self.args.model_name))
            print("\n*****Finish model training*****\n")


    def train(self):
        self.model.train(); self.discriminator.train()
        for idx, input in enumerate(self.train_loader):
            frames, depth, gt_frames, gt_depth, masks, masks_dep = map(lambda x: x.to(self.device), input)
            b, t, c, h, w = frames.size()
            valids = (1 - masks).float()
            depth_loss = 0.
            gt_edge = cal_edge(gt_depth)
            in_edge = gt_edge * valids.squeeze(0)
            pred_img = self.model(frames * valids)
            pred_depth = depth.squeeze(0)

            gt_frames = gt_frames.view(b * t, c, h, w)
            gt_depth = gt_depth.view(b * t, 1, h, w)
            masks = masks.view(b * t, 1, h, w)
            valids = valids.view(b * t, 1, h, w)
            rect_masks = (depth.view(b * t, 1, h, w) > 0).float()
            rected_masks, rected_valids = masks * rect_masks, valids * rect_masks
            if torch.mean(masks) != 0:
                comp_img = gt_frames.view(-1,3,h,w) * valids + masks * pred_img
                hole_loss = self.L1(pred_img * masks, gt_frames * masks)/ torch.mean(masks)
                valid_loss = self.L1(pred_img * valids, gt_frames * valids) / torch.mean(valids)
                perceptual_loss = torch.sum(self.Lpips_func((comp_img+1.)/2., (gt_frames+1.)/2., normalize=True))
            else:
                comp_img = gt_frames.view(-1,3,h,w) * valids + masks * pred_img
                hole_loss = self.L1(pred_img * masks, gt_frames * masks)
                valid_loss = self.L1(pred_img * valids, gt_frames * valids)
                perceptual_loss = torch.sum(self.Lpips_func((comp_img+1.)/2., (gt_frames+1.)/2., normalize=True))
            if torch.mean(rected_masks) != 0:
                depth_loss += 2. * self.L1(gt_depth * rected_masks, pred_depth * rected_masks)/torch.mean(rected_masks)
            else:
                depth_loss += 2. * self.L1(gt_depth * rected_masks, pred_depth * rected_masks)
            if torch.mean(rected_valids) != 0:
                depth_loss += self.L1(pred_depth * rected_valids, gt_depth * rected_valids) / torch.mean(rected_valids)
            else:
                depth_loss += self.L1(pred_depth * rected_valids, gt_depth * rected_valids)

            edge_loss = self.L1(gt_edge, cal_edge(gt_depth * valids + pred_depth * masks))
            if True:
                # discriminator adversarial loss
                real_clip = self.discriminator(gt_frames)
                fake_clip = self.discriminator(comp_img.detach())
                dis_real_loss = self.adversarial_loss(real_clip, True, True)
                dis_fake_loss = self.adversarial_loss(fake_clip, False, True)
                dis_loss = (dis_real_loss + dis_fake_loss) / 2
                self.optimD.zero_grad()
                dis_loss.backward()
                self.optimD.step()
                gen_clip = self.discriminator(comp_img)
                gan_loss = self.adversarial_loss(gen_clip, True, False)

            total_loss = hole_loss + valid_loss +  depth_loss  + perceptual_loss + gan_loss + edge_loss
            total_loss.backward()

            self.optimG.step()
            self.optimG.zero_grad()


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    return rt

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    train()
