import argparse
import json
import os
import math
import matplotlib.pyplot as plt
from patsy.util import widen

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import random
import warnings
import importlib
import numpy as np
import setproctitle
import logging
import datetime

from torch.utils.tensorboard import SummaryWriter

from DataLoader import data_load_index_main

#from train import TrainLoop
from Adv_ST.white_test import TrainLoop
from utils import *



def setup_init(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.benchmark = False   #禁用cuDNN的自动优化 默认False
    th.backends.cudnn.deterministic = True    #启用cuDNN的确定性模式 默认True

def dev(device_id):
    """
    Get the device to use for torch.distributed.
    # """
    if th.cuda.is_available():
        return th.device('cuda:{}'.format(device_id))
    else:
         return th.device("cpu")


def create_argparser():
    defaults = dict(
        patch_size = 2,  #PemsD7L:3  pemsd7:2  PEMS08:1 (Graph:(Metrla:3,pemsd7:2,shenzhen:11))
        seq_len = 24,
        data_dir="",
        weight_decay=0.0,
        early_stop = 10,
        log_interval=1,
        device_id='0',   #设备号
        machine = 'LM1',
        mask_ratio = 0.5,
        random=True,
        eval=True,
        is_block = True,
        t_patch_size = 2,  #TaxiNYCIn:6 Pemsd7:2
        data_norm = 1,
        size = 'middle',
        clip_grad = 0.02, #0.02
        dp_noise_multiplier=5.0,
        mask_strategy = 'causal',  #因果掩码
        mode='training',    #training  | zero |testing |testing_mem
        file_load_path = '',
        dataset = 'GraphPems08_speed_288',  #Pemsd7_288    PemsBay_288  PEMS08_288  TaxiNYCIn_48
        isGraph=True,
        wide=170,  #6    metrla:207,pems04:307,pemsd7:228
        height=1,  #38   65
        his_len = 12,
        pred_len = 12,
        lr_anneal_steps = 150,  #150
        total_epoches = 250,    #200
        lr = 1e-3,  #1e-3  5e-4
        pos_emb = 'SinCos',
        no_qkv_bias = 0,
        is_prompt = 1,
        is_time_emb = 1,
        batch_size_pemsd7 =80,  #32  80  200  400  800(1200s)
        batch_size_PemsBay=80,
        batch_size_pemsd7L=140,
        batch_size_PEMS08=80,
        batch_size_taxibj =60,  #128
        batch_size_nj = 256,
        batch_size_nyc = 160, #512 80
        batch_size_crowd = 256,
        prompt_content = 'node_graph',
        emb_tuning = 0,
        used_data = 'itself',                   #
        num_memory = 512,
        mask_ablation = '',
        data_type = 'GridGraph',
        batch_size_graph_large = 80,  #2.5
        batch_size_graph_small = 64,
        spec_mlp = 0,
        few_ratio = 1.0,
        multi_patch = True,
        seed = 100,
        few_data = '',
        finetune = 0,
        batch_ratio = 0.4,  #批次率
        flag=1, ##
        task_id='clean',  #relaxloss   plain_DP_SGD  Adv-ST offline baseline_new_saliency baseline baseline_new_offline_flag1
        model='UniFlow_A_D_M_F',  #DCN_model  FedFlow_model  UniFlow_Graph UniFlow0
        train_mode='plain',  #   设置：AT_policy_atten:0    AT_policy_atten_dist_offline
        #动态加噪公式
        dp=False,
        cut=0.1,
        balance=0.1,
        # ST_pgd
        test_batch_size=32, #204
        train_num_steps=5,  #5
        train_step_size=0.1,
        train_epsilon= 0.5,
        test_num_steps= 5,
        test_step_size = 0.1,
        test_epsilon= 0.5,
        train_attack_nodes= 0.1,    #0.1
        test_attack_nodes=0.4,      #0.8
        dropout_type='none_dropout',
        attacker='PGD', # ['ALL', 'PGD']
        num_features=1,  # 1
        num_output_features=1,
        num_nodes=170,  #PemsBay:325   PemsD7:228  pems08:170   TaxiNYCIn_48:200
        hidden_embedding_dims=64,
        num_samples=10,
        constant= 0.001,
        is_known_first_node=False,
        baseline='saliency',
        rand_start_step=1,
        distance='l_inf',
        rand_start_mode= 'uniform',
        find_type='random', #貌似无用
        #offline
        policynet_path= 'experiments_all/Exp/GraphPemsd7_288_his_12_pred_12_baseline/model_save/policy_epoch14.pt',
        alpha_reg=0.4,
        resume_epoch=-1,  #上次结束训练的轮次
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



warnings.filterwarnings('ignore', category=FutureWarning)

torch.set_num_threads(32)

def dynamic_model_select(model_name='UniFlow_model'):
    try:
        # 动态导入模块
        module = importlib.import_module(model_name)
        # 从模块中获取 model_select 对象
        model_select = getattr(module, 'model_select')
        return model_select
    except ImportError as e:
        print(f"导入错误: {e}")
    except AttributeError as e:
        print(f"模块中没有 'model_select' 属性: {e}")
    return None


def get_root_logger(args):
    log_level = logging.INFO
    ISOTIMEFORMAT = '%Y.%m.%d-%H.%M.%S'
    thetime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
    os.makedirs(args.model_path, exist_ok=True)

    if 'testing' in args.mode:
        logname = os.path.join(args.model_path, args.mode + '.log')
    else:
        logname = os.path.join(args.model_path, thetime + '_' + args.mode+'_'+args.model+'.log')


    logger = logging.getLogger()
    logger.setLevel(log_level)

    # 先清掉旧的 handlers，避免重复打印
    if logger.hasHandlers():
        logger.handlers.clear()

    fmt = '%(asctime)s - %(levelname)s - %(message)s'
    format_str = logging.Formatter(fmt)

    # 文件日志
    fh = logging.FileHandler(logname, mode='a')
    fh.setFormatter(format_str)
    logger.addHandler(fh)

    # 控制台日志
    sh = logging.StreamHandler()
    sh.setFormatter(format_str)
    logger.addHandler(sh)

    return logger




def main():

    # experiment start
    th.autograd.set_detect_anomaly(False)     #自动梯度异常检测模式 调试使用，训练关闭
    args = create_argparser().parse_args()
    setproctitle.setproctitle("GPU{}".format(args.device_id))
    setup_init(args.seed)
    args.mask_ratio = args.pred_len / (args.pred_len+args.his_len)


    # save path
    if len(args.dataset.split('*'))<10:
        data_replace = args.dataset
    else:
        data_replace = len(args.dataset.split('*'))
    args.folder = '{}_his_{}_pred_{}_{}/'.format(data_replace.replace('*', '_'), args.his_len, args.pred_len,args.task_id)
    model_folder = 'Exp'
    args.model_path = './experiments_all/{}/{}'.format(model_folder,args.folder)
    logdir = "./logs_all/{}/{}".format(model_folder,args.folder)
    if not os.path.exists('./experiments_all/'):
        os.mkdir('./experiments_all/')
    if not os.path.exists('./experiments_all/{}/'.format(model_folder)):
        os.mkdir('./experiments_all/{}/'.format(model_folder))
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
        os.mkdir(args.model_path+'model_save/')

    logger = get_root_logger(args)

    # 保存为JSON格式（需手动处理非JSON兼容类型，如argparse.Namespace）----记录训练的超参数
    if args.mode=='training':    #training
        hyperparams = vars(args)  # 转为字典
        with open(args.model_path + 'train_args.json', 'w') as f:
            json.dump(hyperparams, f, indent=4)  # indent参数美化输出

    logger.info('Start '+args.mode+'....')
    logger.info(args)

    writer = SummaryWriter(log_dir = logdir,flush_secs=5)

    device = dev(args.device_id)

    # load data
    data,  train_index, test_index, val_index, args.scaler = data_load_index_main(args)

    model_func = dynamic_model_select(model_name=args.model)

    model = model_func(args=args)

    if args.multi_patch:
        print('multi_patch')
        model.init_multiple_patch()
    model.init_prompt()


    if 'testing' in args.mode:
        # #加载模型 #src/experiments_all/Exp/Pemsd7_288_his_12_pred_12_plain/model_save/model_best.pt
        model.load_state_dict(torch.load(args.model_path+"/model_save/model_best.pt",map_location=device), strict=True)
        # model.load_state_dict(torch.load("experiments_all/Exp/Pemsd7_288_his_12_pred_12_plain_DP_SGD_bs_32/model_save/model_best.pt",
        #     map_location=device), strict=True)

        logger.info('pretrained model loaded！！！！！！！！！！！！！！！！！！！！！')
    elif args.resume_epoch>-1:
        logger.info('加载模型继续训练epoch:%d',args.resume_epoch+1)
        #model.load_state_dict(torch.load(args.model_path+'model_save/last_epoch{}.pt'.format(args.resume_epoch)))
        model.load_state_dict(torch.load(args.model_path + 'model_save/model_best.pt'))


    model = model.to(device)
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    logger.info('params: {:4f}万'.format(para/1e4))
    # print('params: {:4f}万'.format(para/1e4))
    # print('device:',device)


    # trianing
    args.min_lr = args.lr * 0.1  #修改  0.1

    TrainLoop(
        args = args,
        writer = writer,
        model=model,
        data=data,
        train_index = train_index,
        test_index=test_index, 
        val_index=val_index,
        device=device,
        logger=logger,
        best_rmse = 1e9
    ).run_loop()



if __name__ == "__main__":
    main()