import argparse
import os
import time
import math
import pickle

import numpy as np

import torch
from torch.autograd import Variable
from torchvision import transforms

from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.datasets.datahelpers import cid2filename
from cirtorch.utils.whiten import whitenlearn, whitenapply
from cirtorch.utils.evaluate import compute_map_and_print
from cirtorch.utils.general import get_data_root, htime

datasets_names = {'tumlsi': ['.'], 
                  'cambridge':['ShopFacade', 'KingsCollege', 'StMarysChurch', 'OldHospital'], 
                  '7scenes':['heads', 'chess', 'fire', 'office', 'pumpkin', 'redkitchen', 'stairs']
                  }
whitening_names = ['retrieval-SfM-30k', 'retrieval-SfM-120k']

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Testing')

# network
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--network-path', '-npath', metavar='NETWORK',
                    help='network path, destination where network is saved')
group.add_argument('--network-offtheshelf', '-noff', metavar='NETWORK',
                    help='network off-the-shelf, in the format ARCHITECTURE-POOLING or ARCHITECTURE-POOLING-whiten,' + 
                    ' examples: resnet101-gem | resnet101-gem-whiten')

# test options
parser.add_argument('--data_root', '-dir', metavar='DATA_ROOT', required=True,
                   help='path to the root folder of your datasets')
                   
parser.add_argument('--dataset', '-ds', metavar='DATASETS', default='cambridge', choices=datasets_names,
                   help='name of datasets (default: cambridge)')

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
        for i in range(3):
            f.readline()
        for line in f:
            cur = line.split(' ')
            ims.append(cur[0])
    f.close()
    print('Load images from: {}, total num: {}'.format(fpath, len(ims)))
    return ims
    
def main():
    args = parser.parse_args()
    # setting up the visible GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # loading network from path
    if args.network_path is not None:
        print(">> Loading network:\n>>>> '{}'".format(args.network_path))
        state = torch.load(args.network_path)
        net = init_network(model=state['meta']['architecture'], pooling=state['meta']['pooling'], whitening=state['meta']['whitening'], 
                            mean=state['meta']['mean'], std=state['meta']['std'], pretrained=False)
        net.load_state_dict(state['state_dict'])
        print(">>>> loaded network: ")
        print(net.meta_repr())

    # loading offtheshelf network
    elif args.network_offtheshelf is not None:
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
    # TODO: change to our datasets?
    normalize = transforms.Normalize(
        mean=net.meta['mean'],
        std=net.meta['std']
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # compute whitening
    # TODO: either down load retrieval datasets or change the code to calculate on own datasets
    if args.whitening is not None:
        start = time.time()

        print('>> {}: Learning whitening...'.format(args.whitening))

        # loading db
        db_root = os.path.join(get_data_root(), 'train', args.whitening)
        ims_root = os.path.join(db_root, 'ims')
        db_fn = os.path.join(db_root, '{}-whiten.pkl'.format(args.whitening))
        with open(db_fn, 'rb') as f:
            db = pickle.load(f)
        images = [cid2filename(db['cids'][i], ims_root) for i in range(len(db['cids']))] # Just creating the imlists

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

    for dataset in datasets: 
        start = time.time()

        print('>> {}: Extracting...'.format(dataset))

        # prepare config structure for the test dataset
        images = get_imlist(data_root, dataset, 'dataset_train.txt')
        qimages = get_imlist(data_root, dataset, 'dataset_test.txt')
        
        # extract database and query vectors
        print('>> {}: database images...'.format(dataset))
        vecs = extract_vectors(net, images, args.image_size, transform, root=os.path.join(data_root, dataset), ms=ms, msp=msp)
        print('>> {}: query images...'.format(dataset))
        qvecs = extract_vectors(net, qimages, args.image_size, transform, root=os.path.join(data_root, dataset), ms=ms, msp=msp)
        
        print('>> {}: Evaluating...'.format(dataset))

        # convert to numpy
        vecs = vecs.numpy()
        qvecs = qvecs.numpy()

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
        
        print('>> {}: elapsed time: {}'.format(dataset, htime(time.time()-start)))


if __name__ == '__main__':
    main()

