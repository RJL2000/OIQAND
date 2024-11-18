from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from config import OVO_config
from scipy.optimize import curve_fit
from scipy.special import expit

import numpy as np
import logging
import torch

def mean_squared_error(actual, predicted, squared=True):
    """计算 MSE or RMSE (squared=False)"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # 计算误差
    error = predicted - actual
    
    # 计算均方根误差
    res = np.mean(error**2)
    if squared==False:
        res = np.sqrt(res)
    
    return res


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + expit(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic

def train_ovoiqa(epoch, net, criterion, optimizer, train_loader):
    losses = []
    net.train()
    # save data for one epoch
    pred_all = []
    mos_all = []
    
    # for data in tqdm(train_loader):
    for data in train_loader:
        d = data['d_img_org'].cuda("cuda:0")    # [batch, condition_num, viewports, 3, 224, 224]: [B, 3/1, 8, 3, 224, 224]
        labels = data['score']          # [B, 1]

        labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda("cuda:0")
        pred_d = net(d)     # [B]

        optimizer.zero_grad()
        loss = criterion(torch.squeeze(pred_d), labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        # save results in one epoch
        pred_all = np.append(pred_all, pred_d.data.cpu().numpy())
        mos_all = np.append(mos_all, labels.data.cpu().numpy())
    
    # compute correlation coefficient

    loss = np.mean(losses)

    logistic_pred_all = fit_function(mos_all, pred_all)
    plcc = pearsonr(logistic_pred_all, mos_all)[0]
    srcc = spearmanr(logistic_pred_all, mos_all)[0]
    rmse = mean_squared_error(logistic_pred_all, mos_all, squared=False)


    logging.info('train epoch:{} / loss:{:.4} / PLCC:{:.4}/ SRCC:{:.4}/ RMSE:{:.4} '.format(epoch + 1, loss, plcc, srcc, rmse))

    return loss, plcc, srcc, rmse

def eval_ovoiqa(config, epoch, net, criterion, test_loader):
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_all = []
        mos_all = []

        for data in test_loader:
        # for data in test_loader:
            d = data['d_img_org'].cuda("cuda:0")
            labels = data['score']
            # name = data['name']

            labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda("cuda:0")
            pred_d = net(d)

            # compute loss
            loss = criterion(torch.squeeze(pred_d), labels)
            losses.append(loss.item())

            # save results in one epoch
            pred_all = np.append(pred_all, pred_d.data.cpu().numpy())
            mos_all = np.append(mos_all, labels.data.cpu().numpy())
        
        # compute correlation coefficient
        with open(OVO_config().train_out_file, "w", encoding="utf8") as f:
            f.write("mos,pred\n")
            for i in range(len(np.squeeze(pred_all))):
                f.write(str(np.squeeze(mos_all)[i])+','+str(np.squeeze(pred_all)[i])+'\n')
                

        logistic_pred_all = fit_function(mos_all, pred_all)
        plcc = pearsonr(logistic_pred_all, mos_all)[0]
        srcc = spearmanr(logistic_pred_all, mos_all)[0]
        rmse = mean_squared_error(logistic_pred_all, mos_all, squared=False)

        logging.info('Epoch:{} ===== loss:{:.4} ===== PLCC:{:.4} ===== SRCC:{:.4} ===== RMSE:{:.4}'.format(
                epoch + 1, np.mean(losses), plcc, srcc, rmse))
        return np.mean(losses), plcc, srcc, rmse 