import torch
from torch.utils.data import DataLoader
import os, cv2
import numpy as np
import time
from dataloader.dataset import MVTec
from dsr_model import AnomalyDetetcion, UnetModel
import pandas as pd

def test_recon_device(obj_name, out_path, data_path):

    model = UnetModel(
        in_channels=3,
        out_channels=3,
        base_width=32
    )
    model.cuda()
    model.load_state_dict(
        torch.load(
            rf"D:\Pei-Kai\SCM-guided FAD\models\{obj_name}/checkpoints/70.pckl",
            map_location='cuda:0'))
    model.eval()



    test_dataset = MVTec(rf'D:\pythonProject\AOI\dataset\Classification\YDFID_1', class_name=obj_name, is_train=False,
                          numbers=100)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0)

    recon_path = rf'{out_path}/{obj_name}/recon'
    os.makedirs(recon_path, exist_ok=True)



    for data in test_dataloader:
        imagess, masks, labels, names = data
        images = imagess.cuda()


        ### output
        output_obj = model(images)
        recon_output = output_obj.detach().permute(0, 2, 3, 1).cpu().numpy()
        origin = imagess.permute(0, 2, 3, 1).numpy()[0]
        recon_origin = np.concatenate([recon_output[0], origin], axis=0)
        current_path = rf'{recon_path}/{obj_name}-{labels[0]}-{names[0]}.jpg'
        cv2.imwrite(current_path, recon_origin* 255)


DATA_PATH = r'D:\pythonProject\AOI\dataset\Classification'
OUT_PATH = rf'D:\Pei-Kai\SCM-guided FAD\models'

import argparse

datatype = 'YDFID_1'
normal_data = 'train/defect-free'
obj_batch = [[
    'CL1',
    # 'CL2',
    # 'CL3',
    # 'CL4',
    # 'CL10',
    # 'CL12',
    # 'SL1',
    # 'SL8',
    # 'SL9',
    # 'SL10',
    # 'SL11',
    # 'SL13',
    # 'SL16',
    # 'SP3',
    # 'SP5',
    # 'SP19',
    # 'SP24',
]]



for obj_name in obj_batch:
    print(obj_name[0])
    # model = test_on_device(obj_name,args.data_path, args.out_path, args.lr, args.bs, args.epochs)
    test_recon_device(obj_name[0], OUT_PATH, DATA_PATH)


