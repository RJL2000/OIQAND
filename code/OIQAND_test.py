import os
import torch
import numpy as np
import logging
import torch.nn as nn
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from OIQAND_model import creat_model
from OIQAND_config import config
from OIQAND_dataset import JUFE_10K_Dataset
from tqdm import tqdm
from OIQAND_load_train import fit_function, pearsonr, spearmanr, mean_squared_error


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = config().CUDA_VISIBLE_DEVICES


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

    config = config()

 
    config.log_file = config.model_name + ".log"
    config.ckpt_path = os.path.join(config.ckpt_path, config.type_name, config.model_name)
    config.log_path = os.path.join(config.log_path, config.type_name)

    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
 

    Dataset = JUFE_10K_Dataset
# data load
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


    test_dataset = Dataset(
        info_csv_path= config.test_dataset,
        transform=test_transform
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=8,
        num_workers=config.num_workers,
        drop_last=False,
        shuffle=False
    )
    
    # device_ids = [0, 1,2,3]
    device_ids = config.device_ids
    net = creat_model(config=config, pretrained=config.pretrained)
    net = nn.DataParallel(net, device_ids=device_ids).cuda()

    # test
    net.eval()
    pred_all = []
    mos_all = []
    with torch.no_grad():
        test_loader = tqdm(test_loader)     # 显示进度条
        for data in test_loader:
        # for data in test_loader:
            d = data['d_img_org'].cuda("cuda:0")
            labels = data['score']
            # name = data['name']
            labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda("cuda:0")
            pred_d = net(d)

            # save results in one epoch
            pred_all = np.append(pred_all, pred_d.data.cpu().numpy())
            mos_all = np.append(mos_all, labels.data.cpu().numpy())
        with open(config.test_output_file, "w", encoding="utf8") as f:
            f.write("mos,pred\n")
            for i in range(len(np.squeeze(pred_all))):
                f.write(str(np.squeeze(mos_all)[i])+','+str(np.squeeze(pred_all)[i])+'\n')

        logistic_pred_all = fit_function(mos_all, pred_all)
        plcc = pearsonr(logistic_pred_all, mos_all)[0]
        srcc = spearmanr(logistic_pred_all, mos_all)[0]
        rmse = mean_squared_error(logistic_pred_all, mos_all, squared=False)

        print("plcc: %.4f, srcc: %.4f, rmse: %.4f" % (plcc, srcc, rmse))
