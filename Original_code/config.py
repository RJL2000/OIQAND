import torch


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def OVO_config():

    config = Config({

        "dataset_name": "JUFE-10K",
        
        # optimization
        "batch_size": 16,
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "n_epoch": 25,
        "val_freq": 1,
        "num_workers": 32,
        "split_seed": 0,

        # model
        "num_layers": 6,
        "viewport_nums": 8,
        "embed_dim": 128,
        "dab_layers": 4,

        # 多卡训练
        "CUDA_VISIBLE_DEVICES": "3,4,5",
        "device_ids": [0, 1,2],

        # 是否测试
        "pretrained": True,
        "test_output_file": "./all-JUFE-10K-output_test.csv",

        "train_out_file":"./all-JUFE-10K-output_train.csv",

        # data
        # "model_weight_path": "/media/data/rjl/Assessor360/2D-VR_checkpoints/oiqa.pt",
        "model_weight_path": "/mnt/10T/rjl/12-5-model/pt/all-乱序-8vps/all-乱序-8vps-(12-21-C)/best_ckpt_9.pt",
        "train_dataset": "/mnt/10T/rjl/11_20_train_test_files/train_final_viewport8_mos.csv",
        "test_dataset": "/mnt/10T/rjl/11_20_train_test_files/test_final_viewport8_mos.csv",



        # load & save checkpoint
        "model_name": "11-18-JUFE-10K-vp8",
        "type_name": "all",
        "ckpt_path": "/mnt/10T/rjl/12-5-model/pt",               # directory for saving checkpoint
        "log_path": "/mnt/10T/rjl/12-5-model/output",
    })
    
    
        
        
    return config