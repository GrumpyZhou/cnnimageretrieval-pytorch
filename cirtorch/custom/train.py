import argparse
import os
import math
import numpy as np
import torch 
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.model_zoo import load_url

from cirtorch.custom.dataset import CustomizeTuplesDataset
from cirtorch.datasets.datahelpers import collate_tuples
from cirtorch.networks.imageretrievalnet import init_network
from cirtorch.layers.loss import ContrastiveLoss
from cirtorch.custom.util import split_dataset
from cirtorch.custom.helper import test, train, validate

PRETRAINED = {
    'retrievalSfM120k-vgg16-gem'        : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-vgg16-gem-b4dcdc6.pth',
    'retrievalSfM120k-resnet101-gem'    : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/retrievalSfM120k-resnet101-gem-b80fb85.pth',
}

training_dataset_names = ['7Scenes']
test_datasets = ['chess', 'heads', 'fire', 'office', 'pumpkin', 'redkitchen', 'stairs']
test_datasets = ['heads']
test_whiten_names = ['retrieval-SfM-30k', 'retrieval-SfM-120k']

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
pool_names = ['mac', 'spoc', 'gem', 'rmac']
loss_names = ['contrastive']
optimizer_names = ['sgd', 'adam']

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')

# export directory, training and val datasets, test datasets
parser.add_argument('--data_root', '-dir', metavar='DATA_ROOT', required=True,
                   help='path to the root folder of your datasets')
parser.add_argument('--db-file', '-db', metavar='DB_FILE', required=True,
                   help='path to the db file for data loading')
parser.add_argument('--no-val', dest='val', action='store_false',
                    help='do not run validation')

# network architecture and initialization options
parser.add_argument('--directory', '-odir', metavar='DIR', default='checkpoints/7scenes/',
                    help='destination where trained network should be saved(default:%(default)s)')

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet101', choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: %(default)s)')
parser.add_argument('--pool', '-p', metavar='POOL', default='gem', choices=pool_names,
                    help='pooling options: ' +
                        ' | '.join(pool_names) +
                        ' (default: %(default)s)')
parser.add_argument('--whitening', '-w', dest='whitening', action='store_true',
                    help='train model with end-to-end whitening')
parser.add_argument('--pretrained', dest='pretrained', default='imagenet',
                    help='use model with pretrained weights (default:%(default)s')
parser.add_argument('--loss', '-l', metavar='LOSS', default='contrastive',
                    choices=loss_names,
                    help='training loss options: ' +
                        ' | '.join(loss_names) +
                        ' (default: %(default)s)')
parser.add_argument('--loss-margin', '-lm', metavar='LM', default=0.7, type=float,
                    help='loss margin: (default: %(default)s)')

# train/val options specific for image retrieval learning
parser.add_argument('--image-size', default=362, type=int, metavar='N',
                    help='maximum size of longer image side used for training (default: %(default)s)')
parser.add_argument('--sneg-num', '-snn', default=3, type=int, metavar='N',
                    help='number of negative images from the same cluster as query per train/val tuple (default: %(default)s)')
parser.add_argument('--dneg-num', '-dnn', default=2, type=int, metavar='N',
                    help='number of negative images from different clusters as query per train/val tuple (default: %(default)s)')
parser.add_argument('--query-size', '-qs', default=2000, type=int, metavar='N',
                    help='number of queries randomly drawn per one train epoch (default: %(default)s)')

# standard train/val options
parser.add_argument('--gpu-id', '-g', default='0', metavar='N',
                    help='gpu id used for training (default: 0)')
parser.add_argument('--workers', '-j', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: %(default)s)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run (default: %(default)s)')
parser.add_argument('--batch-size', '-b', default=5, type=int, metavar='N', 
                    help='number of (q,p,n1,...,nN) tuples in a mini-batch (default: %(default)s)')
parser.add_argument('--optimizer', '-o', metavar='OPTIMIZER', default='adam',
                    choices=optimizer_names,
                    help='optimizer options: ' +
                        ' | '.join(optimizer_names) +
                        ' (default: %(default)s)')
parser.add_argument('--lr', '--learning-rate', default=1e-6, type=float,
                    metavar='LR', help='initial learning rate (default: %(default)s)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default:%(default)s)')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: %(default)s)')
parser.add_argument('--resume', default='', type=str, metavar='FILENAME',
                    help='name of the latest checkpoint (default: %(default)s)')

min_loss = float('inf')

def main():
    global args, min_loss
    args = parser.parse_args()

    # create export dir if it doesnt exist
    directory = "_{}".format(args.arch)
    directory += "_{}".format(args.pool)
    if args.whitening:
        directory += "_whiten"
    if args.pretrained in PRETRAINED:
        directory += "_pretrained"
    elif args.pretrained == 'imagenet':
        directory += "_imagenet"
    else:
        directory += "_nopretrain"        
    directory += "_{}_m{:.2f}".format(args.loss, args.loss_margin)
    directory += "_{}_lr{:.1e}_wd{:.1e}".format(args.optimizer, args.lr, args.weight_decay)
    directory += "_snn{}_dnn{}_qsize{}".format(args.sneg_num, args.dneg_num , args.query_size)
    directory += "_bsize{}_imsize{}".format(args.batch_size, args.image_size)

    args.directory = os.path.join(args.directory, directory)
    print(">> Creating directory if it does not exist:\n>> '{}'".format(args.directory))
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)

    # set cuda visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    # set random seeds (maybe pass as argument)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    # create model 
    if args.pretrained:
        if args.pretrained in PRETRAINED:
            # pretrained networks (downloaded automatically)
            print(">> Load pre-trained model from '{}'".format(PRETRAINED[args.pretrained]))
            state = load_url(PRETRAINED[args.pretrained], model_dir=os.path.join('data/networks'))
            model = init_network(model=state['meta']['architecture'], pooling=state['meta']['pooling'], whitening=state['meta']['whitening'], 
                            mean=state['meta']['mean'], std=state['meta']['std'], pretrained=False)
            model.load_state_dict(state['state_dict'])
        else:
            print(">> Using pre-trained model on imagenet '{}'".format(args.arch))
            model = init_network(model=args.arch, pooling=args.pool, whitening=args.whitening, pretrained=True)
    else:
        print(">> Using model from scratch (random weights) '{}'".format(args.arch))
        model = init_network(model=args.arch, pooling=args.pool, whitening=args.whitening, pretrained=False)

    # move network to gpu
    model.cuda()

    # define loss function (criterion) and optimizer
    if args.loss == 'contrastive':
        criterion = ContrastiveLoss(margin=args.loss_margin).cuda()
    else:
        raise(RuntimeError("Loss {} not available!".format(args.loss)))

    # parameters split into features and pool (no weight decay for pooling layer)
    parameters = [
        {'params': model.features.parameters()},
        {'params': model.pool.parameters(), 'lr': args.lr*10, 'weight_decay': 0}
    ]
    if model.whiten is not None:
        parameters.append({'params': model.whiten.parameters()})

    # define optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, args.lr, weight_decay=args.weight_decay)

    # define learning rate decay schedule
    exp_decay = math.exp(-0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay)

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        args.resume = os.path.join(args.directory, args.resume)
        if os.path.isfile(args.resume):
            print(">> Loading checkpoint:\n>> '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            min_loss = checkpoint['min_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(">>>> loaded checkpoint:\n>>>> '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_decay, last_epoch=checkpoint['epoch']-1)
        else:
            print(">> No checkpoint found at '{}'".format(args.resume))

    # Data loading code
    normalize = transforms.Normalize(mean=model.meta['mean'], std=model.meta['std'])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = CustomizeTuplesDataset(
        name='7Scene',
        mode='train', 
        db_file=args.db_file,
        ims_root=args.data_root, 
        imsize=args.image_size, 
        snum=args.sneg_num, 
        dnum=args.dneg_num,
        qsize=args.query_size,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None,
        drop_last=True, collate_fn=collate_tuples
    )
    if args.val:
        val_dataset = CustomizeTuplesDataset(
            name='7Scene',
            mode='val',
            db_file=args.db_file,        
            ims_root=args.data_root,         
            imsize=args.image_size,
            snum=args.sneg_num, 
            dnum=args.dneg_num,
            qsize=float('Inf'),
            transform=transform
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
            drop_last=True, collate_fn=collate_tuples
        )

    # evaluate the network before starting
    gt_root = os.path.join(args.data_root, 'sfm_relative_pose_pairs')
    data_splits = split_dataset(args.data_root, test_datasets, val_step=6, seed=0) 
    test(model, args.data_root, data_splits, gt_root, pass_thres=8, knn=10, query_key='val', db_key='train')

    for epoch in range(start_epoch, args.epochs):

        # set manual seeds per epoch
        np.random.seed(epoch)
        torch.manual_seed(epoch)
        torch.cuda.manual_seed_all(epoch)

        # adjust learning rate for each epoch
        scheduler.step()

        # train for one epoch on train set
        loss = train(train_loader, model, criterion, optimizer, epoch, print_freq=20)

        # evaluate on validation set
        if args.val:
            loss = validate(val_loader, model, criterion, epoch, print_freq=10)

        # evaluate on test datasets
        test(model, args.data_root, data_splits, gt_root, pass_thres=8, knn=10, query_key='val', db_key='train')

        # remember best loss and save checkpoint
        is_best = loss < min_loss
        min_loss = min(loss, min_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'meta': model.meta,
            'state_dict': model.state_dict(),
            'min_loss': min_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.directory)

if __name__ == '__main__':
    main()

