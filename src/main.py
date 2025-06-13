import argparse
import json
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import random
import warnings

import numpy as np
import setproctitle


from torch.utils.tensorboard import SummaryWriter

from DataLoader import data_load_index_main
from DCN_model import model_select


from train import TrainLoop
from utils import *



def setup_init(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True

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
        patch_size = 2,  #unuse
        seq_len = 12,
        data_dir="",
        weight_decay=0.0,
        batch_size=2,  # unuse
        early_stop = 15,
        log_interval=5,
        device_id='0',  #设备号
        machine = 'LM1',
        mask_ratio = 0.5,
        random=True,
        eval=True,
        is_block = True,
        t_patch_size = 3,
        data_norm = 1,
        size = 'middle',
        clip_grad = 0.02, #0.02
        mask_strategy = 'causal',  #因果掩码
        mode='training',    #testing  | zero |training
        file_load_path = '',
        dataset = 'TaxiNYCIn_48',
        his_len = 12,
        pred_len = 12,
        lr_anneal_steps = 1000,  #150
        total_epoches = 5000,   #200
        lr = 1e-3,  #1e-3
        pos_emb = 'SinCos',
        no_qkv_bias = 0,
        is_prompt = 1,
        is_time_emb = 1,
        batch_size_taxibj =64,  #128   越小训练越快
        batch_size_nj = 256,
        batch_size_nyc = 512, #512
        batch_size_crowd = 256,
        prompt_content = 'none',
        emb_tuning = 0,
        used_data = 'itself',                   #？？？？？？
        num_memory = 512,
        mask_ablation = '',
        data_type = 'GridGraph',
        batch_size_graph_large = 128, #3264
        batch_size_graph_small = 64,
        spec_mlp = 0,
        few_ratio = 1.0,
        multi_patch = True,
        seed = 100, 
        few_data = '',
        finetune = 0,
        batch_ratio = 0.4,
        flag=2,
        task_id='DCN_m_GDP_5001',
        dp=True

    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



warnings.filterwarnings('ignore', category=FutureWarning)

torch.set_num_threads(64)  #32

def main():

    # experiment start
    th.autograd.set_detect_anomaly(True)
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

    # 保存为JSON格式（需手动处理非JSON兼容类型，如argparse.Namespace）
    hyperparams = vars(args)  # 转为字典
    with open(args.model_path+'train_args.json', 'w') as f:
        json.dump(hyperparams, f, indent=4)  # indent参数美化输出



    with open(args.model_path+'result_all.txt', 'w') as f:

        f.write('start training\n')


    writer = SummaryWriter(log_dir = logdir,flush_secs=5)

    device = dev(args.device_id)

    # load data
    data,  train_index, test_index, val_index, args.scaler = data_load_index_main(args)

    # # build model
    model = model_select(args=args).to(device)

    if args.multi_patch:
        print('multi_patch')
        model.init_multiple_patch()
    model.init_prompt()

    # # #加载模型
    # model.load_state_dict(torch.load(
    #     "experiments_all/Exp/Size_middle_Dataset_TaxiNYCIn_48_PType_'node_graph'_his_12_pred_12_UseData_'GridGraphall'_machine/model_save/model_best.pth",
    #     map_location=device), strict=True)
    #
    # print('pretrained model loaded！！！！！！！！！！！！！！！！！！！！！！')


    model = model.to(device)
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('params: {:4f}万'.format(para/1e4))
    print('device:',device)


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
        best_rmse = 1e9,
    ).run_loop()



if __name__ == "__main__":
    main()