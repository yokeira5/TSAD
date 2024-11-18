import os
import torch
import numpy as np
import argparse, sys, datetime
from config import Logger
from torch.autograd import Variable
from utils.convert import AdaBN
from utils.memory import Memory
from utils.prompt import Prompt
from utils.metrics import calculate_metrics
from networks.ResUnet_TTA import ResUnet
# from networks.ResUnet import ResUnet
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
from dataloader.dataset import ST
from loss import FocalLoss, BinaryDiceLoss

torch.set_num_threads(1)


class VPTTA:
    def __init__(self, config):
        # Save Log
        time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")
        log_root = os.path.join(config.path_save_log, 'VPTTA')
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        log_path = os.path.join(log_root, time_now + '.log')
        sys.stdout = Logger(log_path, sys.stdout)

        target_test_dataset = ST(rf'{config.dataset_root}', target_list = config.Target_Dataset, is_train=False, numbers=100)

        self.target_test_loader = DataLoader(dataset=target_test_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False,
                                             num_workers=config.num_workers)
        self.image_size = config.image_size

        # Model
        self.load_model = os.path.join(config.model_root, str(config.Source_Dataset))  # Pre-trained Source Model
        self.backbone = config.backbone
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch

        # Optimizer
        self.optim = config.optimizer
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.momentum = config.momentum
        self.betas = (config.beta1, config.beta2)

        # GPU
        self.device = config.device

        # Warm-up
        self.warm_n = config.warm_n

        # Prompt
        self.prompt_alpha = config.prompt_alpha
        self.iters = 5 #config.iters

        # Initialize the pre-trained model and optimizer
        self.build_model()

        # Memory Bank
        self.neighbor = config.neighbor
        self.memory_bank = Memory(size=config.memory_size, dimension=self.prompt.data_prompt.numel())

        # Print Information
        for arg, value in vars(config).items():
            print(f"{arg}: {value}")
        self.print_prompt()
        print('***' * 20)

    def build_model(self):
        self.prompt = Prompt(prompt_alpha=self.prompt_alpha, image_size=self.image_size).to(self.device)
        self.model = ResUnet(resnet=self.backbone, num_classes=self.out_ch, pretrained=False, newBN=AdaBN, warm_n=self.warm_n).to(self.device)
        checkpoint = torch.load(os.path.join(self.load_model, 'last-Res_Unet.pth'))
        self.model.load_state_dict(checkpoint, strict=True)

        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.prompt.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                nesterov=True,
                weight_decay=self.weight_decay
            )
        elif self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.prompt.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay
            )

    def print_prompt(self):
        num_params = 0
        for p in self.prompt.parameters():
            num_params += p.numel()
        print("The number of total parameters: {}".format(num_params))

    def run(self):
        metric_dict = ['Disc_Dice', 'Disc_ASD', 'Cup_Dice', 'Cup_ASD']


        os.makedirs(config.save_root,exist_ok=True)
        # Valid on Target
        metrics_test = [[], [], [], []]

        seg_output1 = []
        for batch, data in tqdm(enumerate(self.target_test_loader), total=len(self.target_test_loader), desc="Processing Batches"):
            x, y, folder_name, file_name = data

            x = x.cuda()
            y = y.cuda()

            self.model.eval()
            self.prompt.train()
            self.model.change_BN_status(new_sample=True)

            # Initialize Prompt
            if len(self.memory_bank.memory.keys()) >= self.neighbor:
                _, low_freq = self.prompt(x)
                init_data, score = self.memory_bank.get_neighbours(keys=low_freq.cpu().numpy(), k=self.neighbor)
            else:
                init_data = torch.ones((1, 1, self.prompt.prompt_size, self.prompt.prompt_size)).data
            self.prompt.update(init_data)

            # Train Prompt for n iters (1 iter in our VPTTA)
            for tr_iter in range(self.iters):
                prompt_x, _ = self.prompt(x)
                self.model(prompt_x)
                times, bn_loss = 0, 0
                for nm, m in self.model.named_modules():
                    if isinstance(m, AdaBN):
                        bn_loss += m.bn_loss
                        times += 1
                loss = bn_loss / times

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.model.change_BN_status(new_sample=False)

            # Inference
            self.model.eval()
            self.prompt.eval()
            with torch.no_grad():
                prompt_x, low_freq = self.prompt(x)
                seg_output1, _, _ = self.model(x)


                seg_output, fea, head_input = self.model(prompt_x)

            # Update the Memory Bank
            self.memory_bank.push(keys=low_freq.cpu().numpy(), logits=self.prompt.data_prompt.detach().cpu().numpy())

            # Calculate the metrics
            # seg_output = torch.sigmoid(pred_logit)

            index_max = np.argmax(seg_output.cpu().detach().numpy(), axis=1)
            index_max1 = np.argmax(seg_output1.cpu().detach().numpy(), axis=1)

            pred_msk = np.uint8(255 * index_max[0])
            pred_msk1 = np.uint8(255 * index_max1[0])
            imgs = np.uint8(255 * x[0, 0, :, :].to('cpu').numpy())
            prompt = np.uint8(255 * prompt_x[0, 0, :, :].to('cpu').numpy())
            imgs = np.concatenate([imgs, prompt, pred_msk1, pred_msk], axis=1)
            cv2.imwrite(os.path.join(config.save_root, f'{config.Source_Dataset}_{folder_name[0]}_{file_name[0]}.jpg'), imgs)

        #     metrics = calculate_metrics(seg_output.detach().cpu(), y.detach().cpu())
        #     for i in range(len(metrics)):
        #         assert isinstance(metrics[i], list), "The metrics value is not list type."
        #         metrics_test[i] += metrics[i]
        #
        # test_metrics_y = np.mean(metrics_test, axis=1)
        # print_test_metric_mean = {}
        # for i in range(len(test_metrics_y)):
        #     print_test_metric_mean[metric_dict[i]] = test_metrics_y[i]
        # print("Test Metrics: ", print_test_metric_mean)
        # print('Mean Dice:', (print_test_metric_mean['Disc_Dice'] + print_test_metric_mean['Cup_Dice']) / 2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Dataset0
    parser.add_argument('--Source_Dataset', type=str, default='train',
                        help='RIM_ONE_r3/REFUGE/ORIGA/REFUGE_Valid/Drishti_GS')
    parser.add_argument('--Target_Dataset', type=list)

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--image_size', type=int, default=256)

    # Model
    parser.add_argument('--backbone', type=str, default='resnet34', help='resnet34/resnet50')
    parser.add_argument('--in_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=2)

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', help='SGD/Adam')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.99)  # momentum in SGD
    parser.add_argument('--beta1', type=float, default=0.9)      # beta1 in Adam
    parser.add_argument('--beta2', type=float, default=0.99)     # beta2 in Adam.
    parser.add_argument('--weight_decay', type=float, default=0.00)

    # Training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--iters', type=int, default=1)

    # Hyperparameters in memory bank, prompt, and warm-up statistics
    parser.add_argument('--memory_size', type=int, default=40)
    parser.add_argument('--neighbor', type=int, default=16)
    parser.add_argument('--prompt_alpha', type=float, default=0.01)
    parser.add_argument('--warm_n', type=int, default=5)

    # Path
    parser.add_argument('--path_save_log', type=str, default='./logs')
    parser.add_argument('--model_root', type=str, default='./models')
    parser.add_argument('--save_root', type=str, default='./results')
    parser.add_argument('--dataset_root', type=str, default=rf'D:\DATASET\resizedata')

    # Cuda (default: the first available device)
    parser.add_argument('--device', type=str, default='cuda:0')

    config = parser.parse_args()

    # ['0Ahuatu', '1065_1', '1065_1_00', '1129_02_00', '1129_1_0407_01', '1129_1_0407_02', '1129_1_0407_03', '1129_2',
    #  '1480_1_0705', '1480_1_0706', '1480_1_0712', '1480_1_0728_15', '1480_1_0728_25', '1480_2_0812', '1480_2_0906',
    #  '1480_2_1', '1480_2_2', '243_1_00', '243_1_20210318_1', '243_1_20210318_2', '243_2_0407', '243_2_0411',
    #  '243_2_0411_2', '839_1_0705', '839_1_0706', '839_1_0712', '839_2', 'Anomaly', 'GRD1065-2022093015-Hole',
    #  'GRD1065-2022093015-Line', 'GRD1065-debug-Hole', 'GRD1065-debug-Line', 'GRD1129-2022093016-Hole',
    #  'GRD1129-2022093016-Line', 'GRD1129-debug-Hole', 'GRD1129-debug-Line', 'GRD243-2022093009-Hole',
    #  'GRD243-2022093009-Line', 'GRD243-debug-Hole', 'GRD243-debug-Line', 'Segment']

    config.Target_Dataset = [
        'GRD1065-2022093015-Hole',
        'GRD1065-2022093015-Line',
        'GRD1129-2022093016-Hole',
        'GRD1129-2022093016-Line',
        'GRD243-2022093009-Hole',
        'GRD243-2022093009-Line',
        rf'Segment\243_02\defect',
        rf'Segment\1480_01\defect'
        # '839_1_0705',
        # '839_2',
        # '1480_1_0728_25',
        # '1480_2_0812',
        # '243_2_0411_2',
    ]
    # config.Target_Dataset.remove(config.Source_Dataset)

    TTA = VPTTA(config)
    TTA.run()
