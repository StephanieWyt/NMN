import tensorflow as tf
import argparse
from include.Config import Config
from include.Model import build, training, get_nbr
from include.Load import *

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

'''
Followed the code style of HGCN-JE-JR:
https://github.com/StephanieWyt/HGCN-JE-JR
'''

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='DBP15k, S-DBP15k or DWY100k')
parser.add_argument('--lang', type=str, help='zh_en, ja_en and fr_en for DBP15K and S-DBP15K, dbp_wd and dbp_yg for DWY100K')

args = parser.parse_args()

if __name__ == '__main__':
    config = Config(args.dataset,args.lang)
    e1 = set(loadfile(config.e1, 1))
    e2 = set(loadfile(config.e2, 1))
    e = len(e1 | e2)

    ILL = loadfile(config.ill, 2)
    illL = len(ILL)
    np.random.shuffle(ILL)
    train = np.array(ILL[:illL // 10 * config.seed])
    test = ILL[illL // 10 * config.seed:]

    KG1 = loadfile(config.kg1, 3)
    KG2 = loadfile(config.kg2, 3)

    output_h, output_h_match, loss_all, sample_w, loss_w, M0, nbr_all, mask_all = \
        build(config.dim, config.dim_g, config.act_func, config.gamma,
            config.k, config.vec, e, 
            config.all_nbr_num, config.sampled_nbr_num, config.beta, KG1 + KG2)
    se_vec, J = training(output_h, output_h_match, loss_all, sample_w, loss_w, config.lr, 
                         config.epochs, config.pre_epochs, train, e,
                         config.k, config.sampled_nbr_num, config.save_suffix, config.dim, config.dim_g, 
                         config.c, config.train_batchnum, config.test_batchnum, 
                         test, M0, e1, e2, nbr_all, mask_all)
    print('loss:', J)
