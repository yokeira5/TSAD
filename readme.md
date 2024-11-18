 train path:
.src\train_ST_segment.py
 


## 需要修改的地方

### train 
_**dataset 路徑: D:\DATASET\resizedata\train_**

`
202行 :
parser.add_argument('--Source_Dataset', type=str, default=rf'train',
                    help='RIM_ONE_r3/REFUGE/ORIGA/REFUGE_Valid/Drishti_GS')
`


`
233行
parser.add_argument('--dataset_root', type=str, default=rf'D:\DATASET\resizedata')
`

### test 
`
train model 的名字
parser.add_argument('--Source_Dataset', type=str, default='train',
                    help='RIM_ONE_r3/REFUGE/ORIGA/REFUGE_Valid/Drishti_GS')
`

`
測試側料列表:
config.Target_Dataset = []
`
