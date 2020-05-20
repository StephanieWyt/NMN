import tensorflow as tf


class Config():
    def __init__(self, d='', l=''):
        dataset = d
        language = l
        if dataset=='S-DBP15k':
            prefix = 'data/DBP15k/' + language 
            self.kg1 = prefix + '/triples_1_s'
            self.kg2 = prefix + '/triples_2_s'
        else:
            prefix = 'data/' + dataset + '/' + language 
            self.kg1 = prefix + '/triples_1'
            self.kg2 = prefix + '/triples_2'
        self.e1 = prefix + '/ent_ids_1'
        self.e2 = prefix + '/ent_ids_2'
        self.ill = prefix + '/ref_ent_ids'
        self.vec = prefix + '/vectorList.json'
        self.save_suffix = dataset+'_'+language
        
        if dataset=='DWY100k':
            self.epochs=200
            self.pre_epochs = 50  # epochs to train the preliminary GCN
            self.train_batchnum=10
            self.test_batchnum=50
            self.all_nbr_num=20
        else:
            self.epochs = 600
            self.pre_epochs = 500
            self.train_batchnum=1
            self.test_batchnum=5
            self.all_nbr_num=100

        self.dim = 300
        self.dim_g = 50
        self.act_func = tf.nn.relu
        self.gamma = 1.0  # margin based loss
        self.k = 125  # number of negative samples for each positive one
        self.seed = 3  # 30% of seeds
        self.c = 20  # size of the candidate set
        self.lr = 0.001

        if dataset=='S-DBP15k':
            if language=='fr_en':
                self.sampled_nbr_num = 10  # number of sampled neighbors
            else:
                self.sampled_nbr_num = 3
            self.beta = 1  # weight of the matching vector
        else:
            self.sampled_nbr_num = 5
            self.beta = 0.1

