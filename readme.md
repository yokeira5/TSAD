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
測試參數修改:
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
`
