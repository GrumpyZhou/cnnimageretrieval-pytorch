import os
import random
import time
import numpy as np
import torch
import shutil
from cirtorch.utils.whiten import whitenlearn, whitenapply

def get_gpu_mem_usage():
    device = torch.cuda.current_device()
    return torch.cuda.memory_allocated(device) / torch.cuda.max_memory_allocated(device) * 100.0

def save_checkpoint(state, is_best, directory):
    filename = os.path.join(directory, 'model_epoch%d.pth.tar' % state['epoch'])
    torch.save(state, filename)
    if is_best:
        filename_best = os.path.join(directory, 'model_best.pth.tar')
        shutil.copyfile(filename, filename_best)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def cal_acc(query_set, db_set, gt_scores, ranks, knn, pass_thres):
    pass_count = 0  # Num of queries with more than pass_thres retrieved images coincide with gts
    total_count = 0  # Num of the counted queries
    sim_scores = []
    coverage = []
    if len(query_set) == len(db_set):
        knn = knn+1 
    for i in range(len(query_set)):
        qim = query_set[i]
        if qim not in gt_scores:
            continue
        total_count += 1
        k = 0 # Num of qualified retrieval ims
        for j in range(knn):
            idx = ranks[j, i]
            rim = db_set[idx]
            if rim == qim: # Skip the most similar match -- itself
                continue
            if rim in gt_scores[qim]['ims']:
                k += 1
                sim_scores.append(gt_scores[qim]['score'][gt_scores[qim]['ims'].index(rim)])
        if k >= pass_thres:
            pass_count += 1
            coverage.append(100.0 * k / knn)
    percent = 100.0 * pass_count / total_count
    coverage = np.mean(coverage)
    mean_sim = np.mean(sim_scores)
    print('Counted: {} Passed: {} Percent: {:.2f}%, Coverage: {:.2f}%,  Mean similarity: {:.4f}'.format(total_count, pass_count, percent, coverage, mean_sim))
    return percent, mean_sim

def eval_retrieval(gt_dir, rank_data, data_splits, pass_thres=5, knn=30, query_key='val', db_key='train'):
    datasets = list(data_splits.keys())
    gt_scores = {}
    avg_percent = []
    avg_sim = []
    for dataset in datasets:
        query_set = data_splits[dataset][query_key]
        db_set = data_splits[dataset][db_key]
        gt_scores[dataset] = {}
        
        # Load ground truth map
        pair_gt_txt = os.path.join(gt_dir, '{}.relative_poses.train.txt'.format(dataset))
        with open(pair_gt_txt, 'r') as f:
            for line in f:
                cur = line.split()
                dbim, qim, score = cur[0], cur[1], float(cur[2]) # TODO: also try reverse order
                if qim not in query_set or dbim not in db_set: 
                    continue
                if qim not in gt_scores[dataset]:
                    gt_scores[dataset][qim] = {'score':[], 'ims':[]}
                gt_scores[dataset][qim]['ims'].append(dbim)
                gt_scores[dataset][qim]['score'].append(score)

        # Evaluate train and test pairs accuracy
        print('>>>Evaluate on {}, query:{}, db: {}, qualified retrieval thres:{}, knn: {}'.format(dataset, query_key, db_key, pass_thres, knn))
        percent, mean_sim = cal_acc(query_set, db_set, gt_scores=gt_scores[dataset], ranks=rank_data[dataset], knn=knn, pass_thres=pass_thres)
        avg_percent.append(percent)
        avg_sim.append(mean_sim)
    avg_percent = np.mean(avg_percent)
    avg_sim = np.mean(avg_sim)
    print('Avg percent: {}, avg similairty {}'.format(avg_percent, avg_sim))
    return avg_percent, avg_sim

def split_dataset(base_dir, datasets, val_step=6, seed=0):
    random.seed(seed)
    print('Val step {}'.format(val_step))
    splitsets = {}
    train_num, val_num = 0, 0
    all_seqs = {}
    for dataset in datasets:
        seq_lines = {}
        splitsets[dataset] = {}
        with open(os.path.join(base_dir, dataset, 'dataset_train.txt'), 'r') as f:
            lines = sorted(f.readlines())
            for line in lines:
                if not line.startswith('seq'):
                    continue
                frame = line.split()[0]
                seq = frame.split('/')[0]
                if seq not in seq_lines:
                    seq_lines[seq] = []
                seq_lines[seq].append(frame)

        val = []
        train = []
        for seq in seq_lines:
            for i,im in enumerate(seq_lines[seq]):
                if i % val_step == 0 and i > 0:
                    val.append(im)
                else:
                    train.append(im)
        print('{} Train: {} Val: {}'.format(dataset, len(train), len(val)))
        splitsets[dataset]['train'] = train
        splitsets[dataset]['val'] = val
        train_num += len(train)
        val_num += len(val)
        all_seqs[dataset] = seq_lines
    print('Train {}  Val {}'.format(train_num, val_num))
    return splitsets

def cal_ranks(vecs, qvecs, Lw):
    # search, rank, and print
    scores = np.dot(vecs.T, qvecs)
    ranks = np.argsort(-scores, axis=0)       

    if Lw is not None:
        # whiten the vectors
        vecs_lw  = whitenapply(vecs, Lw['m'], Lw['P'])
        qvecs_lw = whitenapply(qvecs, Lw['m'], Lw['P'])

        # search, rank, and print
        scores = np.dot(vecs_lw.T, qvecs_lw)
        ranks = np.argsort(-scores, axis=0)
    return scores, ranks

