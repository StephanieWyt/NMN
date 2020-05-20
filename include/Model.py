import math
from .Init import *
import scipy.spatial
import json
import pickle as pkl
import os

def get_vmat(e, KG):
    du = [1] * e
    for tri in KG:
        if tri[0] != tri[2]:
            du[tri[0]] += 1
            du[tri[2]] += 1
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = 1
        else:
            pass
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = 1
        else:
            pass

    return M, du


def get_nbr(M, e, max_nbr):
    nbr = []
    for i in range(e):
        nbr.append([])
    for (i, j) in M:
        if i != j and (max_nbr == -1 or len(nbr[i]) < max_nbr):
            nbr[i].append(j)
    if max_nbr == -1:
        for i in range(e):
            if (len(nbr[i]) > max_nbr):
                max_nbr = len(nbr[i])

    mask = []
    for i in range(e):
        mask.append([1] * len(nbr[i]) + [0] * (max_nbr - len(nbr[i])))
        nbr[i] += [0] * (max_nbr - len(nbr[i]))

    return np.asarray(nbr, dtype=np.int32), np.asarray(mask)


def get_sparse_tensor(e, KG):
    print('getting a sparse tensor...')
    M0, du = get_vmat(e, KG)
    ind = []
    val = []
    for fir, sec in M0:
        ind.append((sec, fir))
        val.append(M0[(fir, sec)] / math.sqrt(du[fir]) / math.sqrt(du[sec]))

    M = tf.SparseTensor(indices=ind, values=val, dense_shape=[e, e])

    return M0, M


def get_se_input_layer(e, dimension, file_path):
    print('adding the primal input layer...')
    with open(file=file_path, mode='r', encoding='utf-8') as f:
        embedding_list = json.load(f)
        print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')
    input_embeddings = tf.convert_to_tensor(embedding_list)
    ent_embeddings = tf.Variable(input_embeddings)
    return tf.nn.l2_normalize(ent_embeddings, 1)


def add_diag_layer(inlayer, dimension, M, act_func, dropout=0.0, init=ones):
    inlayer = tf.nn.dropout(inlayer, 1 - dropout)
    print('adding a layer...')
    w0 = init([1, dimension])
    tosum = tf.sparse_tensor_dense_matmul(M, tf.multiply(inlayer, w0))
    if act_func is None:
        return tosum
    else:
        return act_func(tosum)


def highway(layer1, layer2, dimension):
    kernel_gate = glorot([dimension, dimension])
    bias_gate = zeros([dimension])
    transform_gate = tf.matmul(layer1, kernel_gate) + bias_gate
    transform_gate = tf.nn.sigmoid(transform_gate)
    carry_gate = 1.0 - transform_gate
    return transform_gate * layer2 + carry_gate * layer1


def softmax_positiv(T):
    Tsign = tf.greater(T, 0)
    _reduce_sum = tf.reduce_sum(
        tf.exp(tf.where(Tsign, T, tf.zeros_like(T))), -1, keepdims=True) + math.e
    return tf.where(Tsign, tf.exp(T) / _reduce_sum, T)


def neighborhood_matching(inlayer, mask, max_nbr, beta):
    inlayer_ILL = tf.tile(tf.expand_dims(inlayer[0], 2), [1, 1, max_nbr, 1])
    inlayer_can = tf.tile(tf.expand_dims(inlayer[1], 2), [1, 1, max_nbr, 1])
    inlayer_ILL_trans = tf.transpose(inlayer_ILL, [0, 2, 1, 3])
    inlayer_can_trans = tf.transpose(inlayer_can, [0, 2, 1, 3])
    sim_ILL = tf.reduce_sum(tf.multiply(inlayer_ILL, inlayer_can_trans), -1)
    sim_can = tf.reduce_sum(tf.multiply(inlayer_can, inlayer_ILL_trans), -1)
    mask_ILL = tf.expand_dims(mask[0], -1)
    mask_can = tf.expand_dims(mask[1], 1)
    mask_all = tf.einsum('ijk,ikl->ijl', mask_ILL, mask_can)

    a_ILL = softmax_positiv(tf.multiply(sim_ILL, mask_all))
    a_can = softmax_positiv(tf.multiply(
        sim_can, tf.transpose(mask_all, [0, 2, 1])))

    m_ILL = inlayer[0] - \
        tf.reduce_sum(tf.multiply(inlayer_can_trans,
                                  tf.expand_dims(a_ILL, -1)), 2)
    m_can = inlayer[1] - \
        tf.reduce_sum(tf.multiply(inlayer_ILL_trans,
                                  tf.expand_dims(a_can, -1)), 2)
    m = tf.stack([m_ILL, m_can], 0) * beta
    output_layer = tf.concat([inlayer, m], -1)
    return output_layer


def mock_neighborhood_matching(inlayer, nbr_weight, max_nbr, beta):
    inlayer_ILL = tf.tile(tf.expand_dims(inlayer[0], 2), [1, 1, max_nbr, 1])
    inlayer_can = tf.tile(tf.expand_dims(inlayer[1], 2), [1, 1, max_nbr, 1])
    inlayer_ILL_trans = tf.transpose(inlayer_ILL, [0, 2, 1, 3])
    inlayer_can_trans = tf.transpose(inlayer_can, [0, 2, 1, 3])
    sim_ILL = tf.reduce_sum(tf.multiply(inlayer_ILL, inlayer_can_trans), -1)
    sim_can = tf.reduce_sum(tf.multiply(inlayer_can, inlayer_ILL_trans), -1)
    weight_ILL = tf.expand_dims(nbr_weight[0], -1)
    weight_can = tf.expand_dims(nbr_weight[1], 1)
    weight_all = tf.einsum('ijk,ikl->ijl', weight_ILL, weight_can)

    a_ILL = softmax_positiv(tf.multiply(sim_ILL, weight_all))
    a_can = softmax_positiv(tf.multiply(
        sim_can, tf.transpose(weight_all, [0, 2, 1])))

    m_ILL = inlayer[0] - \
        tf.reduce_sum(tf.multiply(inlayer_can_trans,
                                  tf.expand_dims(a_ILL, -1)), 2)
    m_can = inlayer[1] - \
        tf.reduce_sum(tf.multiply(inlayer_ILL_trans,
                                  tf.expand_dims(a_can, -1)), 2)
    m = tf.stack([m_ILL, m_can], 0) * beta
    output_layer = tf.concat([inlayer, m], -1)
    return output_layer


def neighborhood_aggregation(outlayer, mask, w_gate, w_N, act_func):
    weight_ij = tf.einsum('ijkl,lp->ijkp', outlayer, w_gate)
    if act_func is not None:
        weight_ij = act_func(weight_ij) 
    h_sum = tf.einsum('ijkl,ijkl->ijkl', outlayer, weight_ij)
    h_sum = tf.reduce_sum(tf.multiply(h_sum, tf.expand_dims(mask, -1)), 2)
    h_j = tf.einsum('ijk,kl->ijl', h_sum, w_N) 
    return h_j


def mock_neighborhood_aggregation(outlayer, nbr_weight, w_gate, w_N, act_func):
    weight_ij = tf.einsum('ijkl,lp->ijkp', outlayer, w_gate)
    if act_func is not None:
        weight_ij = act_func(weight_ij)
    h_sum = tf.einsum('ijkl,ijkl->ijkl', outlayer, weight_ij)
    h_sum = tf.reduce_sum(tf.multiply(
        h_sum, tf.expand_dims(nbr_weight, -1)), 2)
    h_j = tf.einsum('ijk,kl->ijl', h_sum, w_N)
    return h_j


def get_loss_pre(outlayer, ILL, gamma, k, neg_left, neg_right, neg2_left, neg2_right):
    left = ILL[:, 0]
    right = ILL[:, 1]
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)

    A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [-1, k])
    D = A + gamma
    L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [-1, 1])))
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg2_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg2_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [-1, k])
    L2 = tf.nn.relu(tf.add(C, tf.reshape(D, [-1, 1])))
    return (tf.reduce_mean(L1) + tf.reduce_mean(L2)) / 2.0


def get_loss_match(outlayer, ILL, gamma, c, dimension):
    out = tf.reshape(outlayer, [2, -1, 2, c, dimension])
    A = tf.reduce_sum(tf.abs(out[0, :, 0, 0] - out[1, :, 0, 0]), -1)
    B = tf.reduce_sum(tf.abs(out[0, :, 0, 1:c] - out[1, :, 0, 1:c]), -1)
    C = - tf.reshape(B, [-1, c - 1])
    D = A + gamma
    L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [-1, 1])))
    B = tf.reduce_sum(tf.abs(out[0, :, 1, 1:c] - out[1, :, 1, 1:c]), -1)
    C = - tf.reshape(B, [-1, c - 1])
    L2 = tf.nn.relu(tf.add(C, tf.reshape(D, [-1, 1])))
    return (tf.reduce_mean(L1) + tf.reduce_mean(L2)) / 2.0


def get_loss_w(select_train, outlayer, nbr_all,
                    mask_all, sample_w, w_gate, w_N,
                    ILL, max_nbr_all, beta):
    left = tf.gather(ILL[:, 0], select_train)
    right = tf.gather(ILL[:, 1], select_train)
    t = 10
    idx = tf.concat([left, right], axis=0) 

    outlayer_idx = tf.gather(outlayer, idx)
    nbr_idx = tf.gather(nbr_all, idx)
    mask_idx = tf.to_float(tf.gather(mask_all, idx))
    outlayer_nbr_idx = tf.gather(outlayer, nbr_idx)
    out_sim = tf.einsum('ij,ijk->ik', tf.matmul(outlayer_idx, sample_w),
                        tf.transpose(outlayer_nbr_idx, [0, 2, 1]))
    nbr_weight = tf.reshape(softmax_positiv(tf.multiply(
        out_sim, mask_idx)), (2, t, -1)) 

    outlayer_idx = tf.reshape(outlayer_idx, (2, t, -1))
    nbr_idx = tf.reshape(nbr_idx, (2, t, -1))
    outlayer_nbr_idx = tf.gather(outlayer, nbr_idx)
    mock_hat_h = mock_neighborhood_matching(
        outlayer_nbr_idx, nbr_weight, max_nbr_all, beta) 
    mock_g = mock_neighborhood_aggregation(
        mock_hat_h, nbr_weight, w_gate, w_N, tf.sigmoid)
    left_x = tf.concat([outlayer_idx[0], mock_g[0]], axis=-1)
    right_x = tf.concat([outlayer_idx[1], mock_g[1]], axis=-1)

    A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
    return tf.reduce_mean(A)


def build(dimension, dimension_g, act_func, gamma, k, vec_path, e, all_nbr_num, sampled_nbr_num, beta, KG):
    tf.reset_default_graph()
    input_layer = get_se_input_layer(e, dimension, vec_path)
    M0, M = get_sparse_tensor(e, KG)
    nbr_all, mask_all = get_nbr(M0, e, all_nbr_num)

    print('KG structure embedding')
    hidden_layer_1 = add_diag_layer(
        input_layer, dimension, M, act_func, dropout=0.0)
    hidden_layer = highway(input_layer, hidden_layer_1, dimension)
    hidden_layer_2 = add_diag_layer(
        hidden_layer, dimension, M, act_func, dropout=0.0)
    output_h = highway(hidden_layer, hidden_layer_2, dimension)
    print('shape of output_h: ', output_h.get_shape())

    c = tf.placeholder(tf.int32, None, "c")
    nbr_sampled = tf.placeholder(tf.int32, [e, sampled_nbr_num], "nbr_sampled")
    mask_sampled = tf.placeholder(tf.float32, [e, sampled_nbr_num], "mask_sampled")
    ILL = tf.placeholder(tf.int32, [None, 2], "ILL")
    candidate = tf.placeholder(tf.int32, [None], "candidate") 
    candidate = tf.reshape(tf.transpose(
        tf.reshape(candidate, (2, -1, c)), (1, 0, 2)), [-1]) 
    idx_pair = tf.stack(
        [tf.reshape(tf.tile(tf.expand_dims(ILL, -1), (1, 1, c)), [-1]), candidate])
    nbr_pair = tf.gather(nbr_sampled, idx_pair)
    mask_pair = tf.gather(mask_sampled, idx_pair)
    h_ctr = tf.nn.embedding_lookup(output_h, idx_pair)
    h_nbr = tf.nn.embedding_lookup(
        output_h, nbr_pair)

    w_gate = glorot([dimension * 2, dimension * 2])
    w_N = glorot([dimension * 2, dimension_g])

    print('neighborhood matching')
    output_hat_h = neighborhood_matching(h_nbr, mask_pair, sampled_nbr_num, beta)
    print('shape of output_hat_h: ', output_hat_h.get_shape())

    print('neighborhood aggregation')
    output_g = neighborhood_aggregation(
        output_hat_h, mask_pair, w_gate, w_N, tf.sigmoid)
    output_h_match = tf.concat([tf.reshape(
        h_ctr, [-1, dimension]), tf.reshape(output_g, [-1, dimension_g])], -1)
    dimension3 = dimension + dimension_g
    print('shape of output_h_match: ', output_h_match.get_shape())

    print("compute pre-training loss")
    neg_left = tf.placeholder(tf.int32, [None], "neg_left") 
    neg_right = tf.placeholder(tf.int32, [None], "neg_right")
    neg2_left = tf.placeholder(tf.int32, [None], "neg2_left")
    neg2_right = tf.placeholder(tf.int32, [None], "neg2_right")
    loss_pre = get_loss_pre(output_h, ILL, gamma, k, neg_left,
                      neg_right, neg2_left, neg2_right)

    print("compute overall loss")
    loss_match = get_loss_match(output_h_match, ILL, gamma, c, dimension3)
    alpha = tf.placeholder(tf.float32, None, "alpha")
    loss_all = (1 - alpha) * loss_pre + alpha * loss_match

    print("compute sampling process loss")
    select_train = tf.placeholder(tf.int32, [10], "select_train")
    sample_w = tf.Variable(tf.eye(dimension, name="sample_w"))
    loss_w = get_loss_w(select_train, output_h, nbr_all,
                             mask_all, sample_w, w_gate, w_N,
                             ILL, all_nbr_num, beta)

    return output_h, output_h_match, loss_all, sample_w, loss_w, M0, nbr_all, mask_all


def get_neg(ILL, output_layer, k, batchnum):
    neg = []
    t = len(ILL)
    ILL_vec = np.array([output_layer[e1] for e1 in ILL])
    KG_vec = np.array(output_layer)
    for p in range(batchnum):
        head = int(t / batchnum * p)
        if p==batchnum-1:
            tail=t
        else:
            tail = int(t / batchnum * (p + 1))
        sim = scipy.spatial.distance.cdist(
            ILL_vec[head:tail], KG_vec, metric='cityblock')
        for i in range(tail - head):
            rank = sim[i, :].argsort()
            neg.append(rank[0: k])

    neg = np.array(neg)
    neg = neg.reshape((t * k,))

    return neg


def np_softmax(x, T=0.1):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x / T) / np.sum(np.exp(x / T), axis=-1, keepdims=True)


def sample_nbr(out, nbr_all, mask_all, e, max_nbr, w, batchnum):
    nbr = []
    for p in range(batchnum):
        head = int(e / batchnum * p)
        if p==batchnum-1:
            tail=e
        else:
            tail = int(e / batchnum * (p + 1))
        mask_p = mask_all[head:tail]
        nbr_p = nbr_all[head:tail]
        sim = np.dot(np.dot(out[head:tail], w), out.transpose())
        x_axis_index = np.tile(np.arange(tail - head),
                               (nbr_all.shape[1], 1)).transpose()

        eps = 1e-8
        prob = sim[x_axis_index, nbr_p] - 1e8 * (1 - mask_p)
        prob = np_softmax(prob) + eps * mask_p
        prob = prob / np.sum(prob, axis=1, keepdims=True)

        for i in range(tail - head):
            if np.sum(mask_p[i]) > max_nbr:
                nbr.append(nbr_p[i, np.random.choice(
                    nbr_all.shape[1], max_nbr, replace=False, p=prob[i])])
            else:
                nbr.append(nbr_p[i, 0:max_nbr])
    mask = mask_all[:, 0:max_nbr]

    return nbr, mask


def mask_candidate(e, e1, e2):
    mask_e1 = np.zeros(e)
    mask_e2 = np.zeros(e)
    for x in e1:
        mask_e1[x] = 1
    for x in e2:
        mask_e2[x] = 1
    return mask_e1, mask_e2


def sample_candidate(ILL, ILL_true, out, k, mask_e, batchnum):
    t = len(ILL)
    e = len(out)
    ILL_vec = np.array([out[x] for x in ILL])
    KG_vec = np.array(out)
    neg = []
    for p in range(batchnum):
        head = int(t / batchnum * p)
        if p==batchnum-1:
            tail=t
        else:
            tail = int(t / batchnum * (p + 1))
        sim = scipy.spatial.distance.cdist(
            ILL_vec[head:tail], KG_vec, metric='cityblock')
        mask_gold = np.zeros((tail - head, e))
        for i in range(tail - head):
            mask_gold[i][ILL_true[i + head]] = 1
        mask = np.tile(mask_e, (tail - head, 1)) + mask_gold
        prob = np_softmax(- sim - 1e8 * mask_gold)
        for i in range(tail - head):
            neg.append(np.random.choice(e, k - 1, replace=False, p=prob[i]))

    candidate = np.concatenate(
        (np.expand_dims(ILL_true, -1), np.asarray(neg)), axis=1)
    candidate = candidate.reshape((t * k,))
    return candidate


def training(output_h, output_h_match, loss_all, sample_w, loss_w, learning_rate, 
             epochs, pre_epochs, ILL, e, k, sampled_nbr_num, save_suffix, dimension, dimension_g, c, 
             train_batchnum, test_batchnum,
             test, M0, e1, e2, nbr_all, mask_all):
    from include.Test import get_hits, get_hits_new
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss_all)
    train_step_w = tf.train.AdamOptimizer(
        learning_rate).minimize(loss_w, var_list=[sample_w])
    print('initializing...')
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print('running...')
    J = []
    ILL = np.array(ILL)
    t = len(ILL)
    ILL_reshape = np.reshape(ILL, 2 * t, order='F')
    L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
    neg_left = L.reshape((t * k,))
    L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
    neg2_right = L.reshape((t * k,))

    nbr_sampled, mask_sampled = get_nbr(M0, e, sampled_nbr_num)
    mask_e1, mask_e2 = mask_candidate(e, e1, e2)
    test_reshape = np.reshape(np.array(test), -1, order='F')
    sample_w_vec = np.identity(dimension)
    test_can_num=50

    if not os.path.exists("model/"):
        os.makedirs("model/")

    if os.path.exists("model/save_"+save_suffix+".ckpt.meta"):
        saver.restore(sess, "model/save_"+save_suffix+".ckpt")
        start_epoch=pre_epochs
    else:
        start_epoch=0

    for i in range(start_epoch, epochs):
        if i % 50 == 0:
            out = sess.run(output_h)
            print('get negative pairs')
            neg2_left = get_neg(ILL[:, 1], out, k, train_batchnum)
            neg_right = get_neg(ILL[:, 0], out, k, train_batchnum)
            print('sample candidates')
            c_left = sample_candidate(ILL[:, 1], ILL[:, 0], out, c, mask_e2, train_batchnum)
            c_right = sample_candidate(ILL[:, 0], ILL[:, 1], out, c, mask_e1, train_batchnum)
            candidate = np.reshape(np.concatenate(
                (c_right, c_left), axis=0), (2, len(ILL), c)) 
            print('sample neighborhood')
            nbr_sampled, mask_sampled = sample_nbr(
                out, nbr_all, mask_all, e, sampled_nbr_num, sample_w_vec, test_batchnum)
            feeddict = {"ILL:0": ILL,
                        "candidate:0": candidate.reshape((-1,)),
                        "neg_left:0": neg_left,
                        "neg_right:0": neg_right,
                        "neg2_left:0": neg2_left,
                        "neg2_right:0": neg2_right,
                        "nbr_sampled:0": nbr_sampled,
                        "mask_sampled:0": mask_sampled,
                        "c:0": c}

            if i < pre_epochs:
                feeddict["alpha:0"] = 0
            else:
                feeddict["alpha:0"] = 1

        for j in range(train_batchnum):
            beg = int(t / train_batchnum * j)
            if j==train_batchnum-1:
                end=t
            else:
                end = int(t / train_batchnum * (j + 1))
            feeddict["ILL:0"] = ILL[beg:end]
            feeddict["candidate:0"] = candidate[:, beg:end].reshape((-1,))
            feeddict["neg_left:0"] = neg_left.reshape(
                (t, k))[beg:end].reshape((-1,))
            feeddict["neg_right:0"] = neg_right.reshape(
                (t, k))[beg:end].reshape((-1,))
            feeddict["neg2_left:0"] = neg2_left.reshape(
                (t, k))[beg:end].reshape((-1,))
            feeddict["neg2_right:0"] = neg2_right.reshape(
                (t, k))[beg:end].reshape((-1,))
            _ = sess.run([train_step], feed_dict=feeddict)

        if i == pre_epochs - 1:
            save_path = saver.save(sess, "model/save_"+save_suffix+".ckpt")
            print("Save to path: ", save_path)

        if i % 10 == 0:
            print('%d/%d' % (i + 1, epochs), 'epochs...')
            outvec = sess.run(output_h, feed_dict=feeddict)
            test_can = get_hits(outvec, test, test_can_num)
            if i >= pre_epochs:
                for j in range(test_batchnum):
                    beg = int(len(test) / test_batchnum * j)
                    if j==test_batchnum-1:
                        end=len(test)
                    else:
                        end = int(len(test) / test_batchnum * (j + 1))
                    feeddict_test = {"ILL:0": test[beg:end],
                                     "candidate:0": test_can[:, beg:end].reshape((-1,)),
                                     "nbr_sampled:0": nbr_sampled,
                                     "mask_sampled:0": mask_sampled,
                                     "c:0": test_can_num}
                    outvec_h_match = sess.run(
                        output_h_match, feed_dict=feeddict_test)
                    if j == 0:
                        outvec_h_match_all = outvec_h_match.reshape((2, -1, dimension+dimension_g))
                    else:
                        outvec_h_match_all = np.concatenate(
                            [outvec_h_match_all, outvec_h_match.reshape((2, -1, dimension+dimension_g))], axis=1)
                get_hits_new(outvec_h_match_all, test_can, test, test_can_num)

        if i >= pre_epochs and i % 50 == 49:
            print('train sample w')
            for _ in range(10):
                select_train = np.random.choice(len(ILL), 10)
                feeddict["select_train:0"] = select_train
                for j in range(5):
                    _, thw = sess.run([train_step_w, loss_w],
                                      feed_dict=feeddict)
                print(thw)
            sample_w_vec = sess.run(sample_w, feed_dict=feeddict)

    sess.close()
    return outvec, J
