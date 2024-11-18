import os
import torch
import numpy as np
import argparse, sys, datetime

from tqdm import tqdm

from config import *
from torchnet import meter
from networks.ResUnet import ResUnet
from torch.utils.data import DataLoader
from utils.metrics import calculate_metrics
from dataloaders.OPTIC_dataloader import OPTIC_dataset
from dataloaders.convert_csv_to_list import convert_labeled_list
from dataloaders.normalize import normalize_image, normalize_image_to_0_1
from dataloaders.transform import collate_fn_wo_transform, collate_fn_w_transform
from dataloader.dataset import ST, ST_double
from loss import FocalLoss, BinaryDiceLoss
import cv2

class TrainSource:
    def __init__(self, config):
        # Save Log and Model
        time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")
        self.model_path = os.path.join(config.path_save_model, config.Source_Dataset)  # Save Model
        self.log_path = os.path.join(config.path_save_log, 'train_Source')  # Save Log
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.log_path = os.path.join(self.log_path, time_now + '.log')
        sys.stdout = Logger(self.log_path, sys.stdout)

        # Data Loading

        train_dataset = ST_double(rf'{config.dataset_root}/{config.Source_Dataset}', is_train=True, numbers=100)
        print('Source Train Dataset: ', train_dataset, len(train_dataset))

        self.source_train_loader = DataLoader(dataset=train_dataset,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              # pin_memory=True,
                                              # collate_fn=collate_fn_w_transform,
                                              num_workers=config.num_workers)
        self.image_size = config.image_size

        # Model
        self.backbone = config.backbone
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch

        # Loss
        self.lossmap = config.lossmap
        self.seg_cost = Seg_loss(self.lossmap)

        # Optimizer
        self.optim = config.optimizer
        self.lr_scheduler = config.lr_scheduler
        self.lr = config.lr
        self.momentum = config.momentum
        self.weight_decay = config.weight_decay
        self.betas = (config.beta1, config.beta2)

        # Training
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size

        # GPU
        self.device = config.device

        # Initialize the pre-trained model and optimizer
        self.build_model()

        # Print Information
        for arg, value in vars(config).items():
            print(f"{arg}: {value}")
        self.print_network()
        print('***' * 20)

    def build_model(self):
        self.model = ResUnet(resnet=self.backbone, num_classes=self.out_ch, pretrained=True).to(self.device)

        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=True
            )
        elif self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                betas=self.betas
            )
        elif self.optim == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                betas=self.betas
            )

        if self.lr_scheduler == 'Cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50, eta_min=1e-7)
        elif self.lr_scheduler == 'Step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)
        elif self.lr_scheduler == 'Epoch':
            self.scheduler = EpochLR(self.optimizer, epochs=self.num_epochs, gamma=0.9)
        else:
            self.scheduler = None

    def print_network(self):
        num_params = 0
        for p in self.model.parameters():
            num_params += p.numel()
        print("The number of total parameters: {}".format(num_params))

    def run(self):
        metrics_test = [[], [], [], []]
        metric_dict = ['Disc_Dice', 'Disc_ASD', 'Cup_Dice', 'Cup_ASD']
        best_loss, best_epoch = np.inf, 0
        loss_meter = meter.AverageValueMeter()

        loss_focal = FocalLoss()
        loss_dice = BinaryDiceLoss()

        os.makedirs(rf'{self.model_path}/train', exist_ok=True)

        file_name = ''
        seg_output = ''
        x = ''

        for epoch in tqdm(range(self.num_epochs), total=self.num_epochs, desc="Processing Training Epoch"):
            self.model.train()
            print("Epoch:{}/{}".format(epoch + 1, self.num_epochs))
            print("Source Pretraining...")
            print("Learning rate: " + str(self.optimizer.param_groups[0]["lr"]))
            loss_meter.reset()

            for batch, data in enumerate(self.source_train_loader):
                x_ori, x, y, file_name, ws = data

                x = x.cuda()
                y = y.cuda()

                y[y > 0.5] = 1
                y[y <= 0.5] = 0

                # 將 mask 轉為整數類型
                y = y.long()  # 將布爾型或浮點型轉換為整數類型
                # 移除第 0 維並轉為 one-hot 編碼
                mask_one_hot = torch.nn.functional.one_hot(y.squeeze(1), num_classes=2)  # 形狀變為 (256, 256, 2)

                # 轉置維度，使最終形狀為 (2, 256, 256)
                mask_one_hot = mask_one_hot.permute(0, 3, 1, 2)
                #
                # mask = y.squeeze(1)

                seg_output, fea = self.model(x)
                # loss = loss_focal(seg_output, mask)
                # loss_ce = self.seg_cost(seg_output, mask_one_hot)
                loss = self.seg_cost(seg_output, mask_one_hot.float())
                 # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_meter.add(loss.item())
                metrics = calculate_metrics(seg_output.detach().cpu(), mask_one_hot.detach().cpu())
                for i in range(len(metrics)):
                    assert isinstance(metrics[i], list), "The metrics value is not list type."
                    metrics_test[i] += metrics[i]

            if epoch % 10 == 0:
                index_max = np.argmax(seg_output.cpu().detach().numpy(), axis=1)
                pred_msk = np.uint8(255 * index_max[0])
                imgs = np.uint8(255 * x[0, 0, :, :].to('cpu').numpy())
                imgs = np.concatenate([imgs, pred_msk], axis=1)
                cv2.imwrite(rf'{self.model_path}/train/{epoch}_{file_name[0]}.jpg', imgs)

            # if self.scheduler is not None:
            #     self.scheduler.step()

            print("Train ———— Total Loss:{:.8f}".format(loss_meter.value()[0]))
            metrics_y = np.mean(metrics_test, axis=1)
            print_test_metric = {}
            for i in range(len(metrics_y)):
                print_test_metric[metric_dict[i]] = metrics_y[i]
            print("Train Metrics Mean: ", print_test_metric)
            print('*****'*10)

            # Save Model
            if best_loss > loss_meter.value()[0]:
                best_loss = loss_meter.value()[0]
                best_epoch = (epoch + 1)
                torch.save(self.model.state_dict(), self.model_path + '/' + 'pretrain-Res_Unet.pth')

        torch.save(self.model.state_dict(), self.model_path + '/' + 'last-Res_Unet.pth')
        print('The best total loss:{} epoch:{}'.format(best_loss, best_epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--Source_Dataset', type=str, default=rf'train',
                        help='RIM_ONE_r3/REFUGE/ORIGA/REFUGE_Valid/Drishti_GS')

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--image_size', type=int, default=256)

    # Model
    parser.add_argument('--backbone', type=str, default='resnet34', help='resnet34/resnet50')
    parser.add_argument('--in_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=2)

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='AdamW', help='SGD/Adam/AdamW')
    parser.add_argument('--lr_scheduler', type=str, default='Epoch',
                        help='Cosine/Step/Epoch')   # choose the decrease strategy of lr
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.001)  # weight_decay in SGD
    parser.add_argument('--momentum', type=float, default=0.99)  # momentum in SGD
    parser.add_argument('--beta1', type=float, default=0.9)  # beta1 in Adam
    parser.add_argument('--beta2', type=float, default=0.99)  # beta2 in Adam

    # Training
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)

    # Loss function
    parser.add_argument('--lossmap', type=str, default=['dice', 'bce'])

    # Path
    parser.add_argument('--path_save_log', type=str, default='./logs/')
    parser.add_argument('--path_save_model', type=str, default='./models/')
    parser.add_argument('--dataset_root', type=str, default=rf'D:\DATASET\resizedata')
    # parser.add_argument('--dataset_root', type=str, default='/media/userdisk0/zychen/Datasets/Fundus')

    # Cuda (default: the first available device)
    parser.add_argument('--device', type=str, default='cuda:0')

    config = parser.parse_args()

    TS = TrainSource(config)
    TS.run()
