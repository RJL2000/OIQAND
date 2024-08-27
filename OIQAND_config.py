
class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def config():

    config = Config({

        "dataset_name": "JUFE_10K",
        
        # optimization
        "batch_size": 8,
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "n_epoch": 25,
        "val_freq": 1,
        "num_workers": 32,
        "split_seed": 0,

        # model
        "viewport_nums": 8,
        "embed_dim": 128,

        # 多卡训练
        "CUDA_VISIBLE_DEVICES": "2, 3, 4",
        "device_ids": [0, 1, 2],

        # 是否测试
        "pretrained": False,

        # 是否生成预测文件
        "print_pred_file": False,
        "test_output_file": "./pre_output_test.csv",

        # data
        # "model_weight_path": "/media/data/rjl/Assessor360/2D-VR_checkpoints/oiqa.pt",
        "model_weight_path": "",
        #vp8
        # "train_dataset": "/home/d310/10t/rjl/TMM_OIQA/file/JUFE-10K/train_final_viewport8_mos.csv",
        # "test_dataset": "/home/d310/10t/rjl/TMM_OIQA/file/JUFE-10K/test_final_viewport8_mos.csv",
        #vp20
        "train_dataset": "/home/d310/10t/rjl/TMM_OIQA/file/JUFE-10K/train_CNN_viewport8_chidao.csv",
        "test_dataset": "/home/d310/10t/rjl/TMM_OIQA/file/JUFE-10K/test_CNN_viewport8_chidao.csv",



        # load & save checkpoint
        "model_name": "OIQAND_vp8_cnn_oiqa_chidao",
        "type_name": "all",
        "ckpt_path": "/home/d310/10t/rjl/TMM_OIQA/pt",               # directory for saving checkpoint
        "log_path": "/home/d310/10t/rjl/TMM_OIQA/output",
    })
    
    
    return config