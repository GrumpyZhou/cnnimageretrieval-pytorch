import argparse
import os
import time
import math
import pickle

import numpy as np

import torch
from torch.utils.model_zoo import load_url
from torch.autograd import Variable
from torchvision import transforms

from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.datasets.datahelpers import cid2filename
from cirtorch.utils.whiten import whitenlearn, whitenapply
from cirtorch.utils.evaluate import compute_map_and_print
from cirtorch.utils.general import get_data_root, htime

PRETRAINED = {
    'retrievalSfM120k-vgg16-gem'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'retrievalSfM120k-resnet101-gem'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
}

datasets_names = {'tumlsi': ['.'], 
                  'cambridge':['ShopFacade', 'KingsCollege', 'StMarysChurch', 'OldHospital'], 
                  '7scenes':['heads', 'chess', 'fire', 'office', 'pumpkin', 'redkitchen', 'stairs']
                  }
whitening_names = ['retrieval-SfM-30k', 'retrieval-SfM-120k']

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Testing')

# Network
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--network-path', '-npath', metavar='NETWORK',
                    help='network path, destination where network is saved')
group.add_argument('--network-offtheshelf', '-noff', metavar='NETWORK',
                    help='network off-the-shelf, in the format ARCHITECTURE-POOLING or ARCHITECTURE-POOLING-whiten,' + 
                    ' examples: resnet101-gem | resnet101-gem-whiten')

# Datset options
parser.add_argument('--data_root', '-dir', metavar='DATA_ROOT', required=True,
                   help='path to the root folder of your datasets')
parser.add_argument('--dataset', '-ds', metavar='DATASETS', default='cambridge', choices=datasets_names,
                   help='name of datasets (default: cambridge)')
parser.add_argument('--train_txt', '-txt', metavar='TRAIN_TXT', default='dataset_train.txt', 
                    help='file to load train images')
parser.add_argument('--query_txt', '-qtxt', metavar='QUERY_TXT', default='dataset_test.txt', 
                   help='file to load query images')
parser.add_argument('--outfile', '-f', metavar='OUTFILE', default='cambridge.npy',
                   help='name of the output file where retrieval results are stored')

# Testing options
parser.add_argument('--image-size', '-imsize', default=None, type=int, metavar='N',
                    help='maximum size of longer image side used for testing (default: None)')
parser.add_argument('--multiscale', '-ms', dest='multiscale', action='store_true',
                    help='use multiscale vectors for testing')
parser.add_argument('--whitening', '-w', metavar='WHITENING', default=None, choices=whitening_names,
                    help='dataset used to learn whitening for testing: ' + 
                        ' | '.join(whitening_names) + 
                        ' (default: None)')

# GPU ID
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help='gpu id used for testing (default: 0)')

def get_imlist(data_root, dataset, ftxt):
    ims = []
    fpath = os.path.join(data_root, dataset, ftxt)
    with open(fpath) as f:
        for line in f:
            if not line.startswith('seq') and not line.startswith('00'):
                continue
            cur = line.split(' ')
            ims.append(cur[0])
    f.close()
    print('Load images from: {}, total num: {}'.format(fpath, len(ims)))
    return ims
    
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

def main():
    args = parser.parse_args()
    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    # loading network from path
    result_dir = 'retreival_results'
    if args.network_path is not None:
        result_dir = os.path.join(result_dir, args.network_path)
        print(">> Loading network:\n>>>> '{}'".format(args.network_path))
        if args.network_path in PRETRAINED:
            # pretrained networks (downloaded automatically)
            state = load_url(PRETRAINED[args.network_path], model_dir=os.path.join(get_data_root(), 'networks'))
        else:
            state = torch.load(args.network_path)
        net = init_network(model=state['meta']['architecture'], pooling=state['meta']['pooling'], whitening=state['meta']['whitening'], 
                            mean=state['meta']['mean'], std=state['meta']['std'], pretrained=False)
        net.load_state_dict(state['state_dict'])
        
        # if whitening is precomputed
        if 'Lw' in state['meta']:
            net.meta['Lw'] = state['meta']['Lw']
        
        print(">>>> loaded network: ")
        print(net.meta_repr())

    # loading offtheshelf network
    elif args.network_offtheshelf is not None:
        result_dir = os.path.join(result_dir, args.network_offtheshelf)        
        offtheshelf = args.network_offtheshelf.split('-')
        if len(offtheshelf)==3:
            if offtheshelf[2]=='whiten':
                offtheshelf_whiten = True
            else:
                raise(RuntimeError("Incorrect format of the off-the-shelf network. Examples: resnet101-gem | resnet101-gem-whiten"))
        else:
            offtheshelf_whiten = False
        print(">> Loading off-the-shelf network:\n>>>> '{}'".format(args.network_offtheshelf))
        net = init_network(model=offtheshelf[0], pooling=offtheshelf[1], whitening=offtheshelf_whiten)
        print(">>>> loaded network: ")
        print(net.meta_repr())

    # setting up the multi-scale parameters
    ms = [1]
    msp = 1
    if args.multiscale:
        ms = [1, 1./math.sqrt(2), 1./2]
        if net.meta['pooling'] == 'gem' and net.whiten is None:
            msp = net.pool.p.data.tolist()[0]

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()
    
    # set up the transform
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # compute whitening
    if args.whitening is not None:
        start = time.time()
        if 'Lw' in net.meta and args.whitening in net.meta['Lw']:
            
            print('>> {}: Whitening is precomputed, loading it...'.format(args.whitening))
            
            if args.multiscale:
                Lw = net.meta['Lw'][args.whitening]['ms']
            else:
                Lw = net.meta['Lw'][args.whitening]['ss']
        else:
            # Save whitening TODO
            print('>> {}: Learning whitening...'.format(args.whitening))

            # loading db
            db_root = os.path.join(get_data_root(), 'train', args.whitening)
            ims_root = os.path.join(db_root, 'ims')
            db_fn = os.path.join(db_root, '{}-whiten.pkl'.format(args.whitening))
            with open(db_fn, 'rb') as f:
                db = pickle.load(f)
            images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))]

            # extract whitening vectors
            print('>> {}: Extracting...'.format(args.whitening))
            wvecs = extract_vectors(net, images, args.image_size, transform, ms=ms, msp=msp)
            
            # learning whitening 
            print('>> {}: Learning...'.format(args.whitening))
            wvecs = wvecs.numpy()
            m, P = whitenlearn(wvecs, db['qidxs'], db['pidxs'])
            Lw = {'m': m, 'P': P}

        print('>> {}: elapsed time: {}'.format(args.whitening, htime(time.time()-start)))
    else:
        Lw = None

    # evaluate on test datasets
    data_root = args.data_root
    datasets = datasets_names[args.dataset]
    result_dict = {}
    for dataset in datasets: 
        start = time.time()
        result_dict[dataset]= {}
        print('>> {}: Extracting...'.format(dataset))
        
        # prepare config structure for the test dataset
        images = get_imlist(data_root, dataset, args.train_txt)
        qimages = get_imlist(data_root, dataset, args.query_txt)
        
        # extract database and query vectors
        print('>> {}: database images...'.format(dataset))
        vecs = extract_vectors(net, images, args.image_size, transform, root=os.path.join(data_root, dataset), ms=ms, msp=msp)
        print('>> {}: query images...'.format(dataset))
        qvecs = extract_vectors(net, qimages, args.image_size, transform, root=os.path.join(data_root, dataset), ms=ms, msp=msp)
        print('>> {}: Evaluating...'.format(dataset))

        # convert to numpy
        vecs = vecs.numpy()
        qvecs = qvecs.numpy()
        scores, ranks = cal_ranks(vecs, vecs, Lw)
        result_dict[dataset]['train'] = {'scores':scores, 'ranks':ranks}
        scores, ranks = cal_ranks(vecs, qvecs, Lw)        
        result_dict[dataset]['test']  = {'scores':scores, 'ranks':ranks}
        print('>> {}: elapsed time: {}'.format(dataset, htime(time.time()-start)))

    # Save retrieval results
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_file = os.path.join(result_dir, args.outfile)
    np.save(result_file, result_dict)
    print('Save retrieval results to {}'.format(result_file))

if __name__ == '__main__':
    main()

