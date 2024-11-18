import os
import torch
import numpy as np
import logging
import time
import torch.nn as nn
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from OVOIQ import creat_model
from config import OVO_config
from torch.utils.tensorboard import SummaryWriter 
from load_train import train_ovoiqa, eval_ovoiqa
from my_dataset import O10K_Dataset


# os.environ['NCCL_DEBUG'] = 'INFO'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = OVO_config().CUDA_VISIBLE_DEVICES
# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logging(config):
    if not os.path.exists(config.log_path): 
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )

if __name__ == '__main__':
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)  

    setup_seed(20)

    config = OVO_config()

 
    config.log_file = config.model_name + ".log"
    config.ckpt_path = os.path.join(config.ckpt_path, config.type_name, config.model_name)
    config.log_path = os.path.join(config.log_path, config.type_name)

    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
 

    set_logging(config)
    logging.info(config)

    Dataset = O10K_Dataset

    # data load
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = Dataset(
        info_csv_path= config.train_dataset,
        transform=train_transform
    )
    val_dataset = Dataset(
        info_csv_path= config.test_dataset,
        transform=test_transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=False,
        shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        drop_last=False,
        shuffle=False
    )
    
    # device_ids = [0, 1,2,3]
    device_ids = config.device_ids
    net = creat_model(config=config, pretrained=False)
    net = nn.DataParallel(net, device_ids=device_ids).cuda()

    # loss function
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    # train & validation
    losses, scores = [], []
    best_srocc = 0
    best_plcc = 0
    main_score = 0

    for epoch in range(0, config.n_epoch):
        # visual(net, val_loader)
        start_time = time.time()
        logging.info('Running training epoch {}'.format(epoch + 1))
        loss_val, plcc, srcc, rmse = train_ovoiqa(epoch, net, criterion, optimizer, train_loader)
        print("[train epoch %d/%d] loss: %.6f, plcc: %.4f, srcc: %.4f, rmse: %.4f, lr: %.6f, time: %.2f min" % \
                (epoch+1, config.n_epoch, loss_val, plcc, srcc, rmse, optimizer.param_groups[0]["lr"], (time.time() - start_time) / 60))
        
        if (epoch + 1) % config.val_freq == 0:
            start_time = time.time()
            logging.info('Starting eval...')
            logging.info('Running val {} in epoch {}'.format(config.dataset_name, epoch + 1))
            loss, plcc_v, srcc_v, rmse_v = eval_ovoiqa(config, epoch, net, criterion, val_loader)
            print("[val epoch %d/%d] loss: %.6f, plcc_v: %.4f, srcc_v: %.4f, rmse_v: %.4f, lr: %.6f, time: %.2f min" % \
                (epoch+1, config.n_epoch, loss, plcc_v, srcc_v, rmse_v, optimizer.param_groups[0]["lr"], (time.time() - start_time) / 60))
          
            if plcc_v + srcc_v > main_score:
                main_score = plcc_v + srcc_v
                best_srcc = srcc_v
                best_plcc = plcc_v
                # save weights
                # ckpt_name = "best_ckpt.pt"
                model_save_path = os.path.join(config.ckpt_path, "best_ckpt_"+str(epoch+1)+".pt")
                torch.save(net.module.state_dict(), model_save_path)

    