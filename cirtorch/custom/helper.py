import os
import time
import numpy as np
import torch 
import torchvision.transforms as transforms
from cirtorch.networks.imageretrievalnet import extract_vectors
from cirtorch.custom.util import * 

def set_batchnorm_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        # freeze running mean and std
        m.eval()
        # freeze parameters
        # for p in m.parameters():
            # p.requires_grad = False

def train(train_loader, model, criterion, optimizer, epoch, print_freq=40, log=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # create tuples for training
    avg_dist = train_loader.dataset.create_epoch_tuples(model)
    lprint('>>>>Train Average negative distance: {:.2f}'.format(avg_dist), log)
    
    # switch to train mode
    model.train()
    model.apply(set_batchnorm_eval)
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # zero out gradients so we can accumulate new ones over batches
        optimizer.zero_grad()

        nq = len(input) # number of training tuples
        ni = len(input[0]) # number of images per tuple

        for q in range(nq):
            output = torch.autograd.Variable(torch.Tensor(model.meta['outputdim'], ni).cuda())
            for imi in range(ni):
                # target = target.cuda(async=True)
                input_var = torch.autograd.Variable(input[q][imi].cuda())

                # compute output
                output[:, imi] = model(input_var).squeeze()

            # compute loss for this batch and do backward pass for a batch
            # each backward pass gradients will be accumulated
            target_var = torch.autograd.Variable(target[q].cuda())
            loss = criterion(output, target_var)
            losses.update(loss.item())
            loss.backward()

        # do one step for multiple batches
        # accumulated gradients are used
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('>> Train: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'gpu memory usage {usage:.2f}%'.format(
                   epoch+1, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, usage=get_gpu_mem_usage()))
    lprint('>>Epoch {} finally training acc: Avg loss: {:.4f}, Avg time: {:.2f}'.format(epoch+1, losses.avg, batch_time.avg), log)
    return losses.avg

def validate(val_loader, model, criterion, epoch, print_freq=100, log=None):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # create tuples for validation
    avg_dist = val_loader.dataset.create_epoch_tuples(model)
    lprint('>>>>Val Average negative distance: {:.2f}'.format(avg_dist), log)
    
    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():    
        for i, (input, target) in enumerate(val_loader):
            nq = len(input) # number of training tuples
            ni = len(input[0]) # number of images per tuple
            output = torch.Tensor(model.meta['outputdim'], nq*ni).cuda()

            for q in range(nq):
                for imi in range(ni):
                    # target = target.cuda(async=True)
                    input_var = input[q][imi].cuda()

                    # compute output
                    output[:, q*ni + imi] = model(input_var).squeeze()

            target_var = torch.cat(target).cuda()
            loss = criterion(output, target_var)

            # record loss
            losses.update(loss.item()/nq, nq)

            # measure elapsed time
            batch_time.update(time.time() - end)

            if i % print_freq == 0:
                print('>> Val: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'gpu memory usage {usage:.2f}%'.format(
                      epoch+1, i, len(val_loader), batch_time=batch_time,
                      loss=losses, usage=get_gpu_mem_usage()))
    lprint('>>Epoch {} finally validation acc: Avg loss: {:.4f}, Avg time: {:.2f}'.format(epoch+1, losses.avg, batch_time.avg), log)
    return losses.avg


def test(net, data_root, data_splits, gt_root, epoch, pass_thres=8, knn=10, query_key='val', db_key='train', log=None):
    print('>> Evaluating network on test datasets...')

    # for testing we use image size of max 1024
    image_size = 1024

    # moving network to gpu and eval mode
    net.cuda()
    net.eval()
    
    # Data loading code
    normalize = transforms.Normalize(mean=net.meta['mean'], std=net.meta['std'])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # TODO: Whitening
    rank_data = {}
    for dataset in data_splits: 
        print('>> Evaluate model on {}'.format(dataset))
        start = time.time()
        dbims = data_splits[dataset][db_key]
        qims = data_splits[dataset][query_key]
        dbvecs = extract_vectors(net, dbims, image_size, transform, root=os.path.join(data_root, dataset))
        qvecs = extract_vectors(net, qims, image_size, transform, root=os.path.join(data_root, dataset))
        print('>> Extracted database images: {} query images: {} time: {:.2f} gpu usage: {:.2f}%'.format(dbvecs.size(), qvecs.size(), time.time()-start, get_gpu_mem_usage()))

        # Retrieval
        dbvecs = dbvecs.numpy()
        qvecs = qvecs.numpy()
        scores, ranks = cal_ranks(dbvecs, qvecs, Lw=None)        
        rank_data[dataset] = ranks
    print('>> Total elapsed time for retrieval: {:.2f}'.format(time.time()-start))

    # Calculate accuracy with gt_score
    start = time.time()
    avg_percent, avg_sim = eval_retrieval(gt_root, rank_data, data_splits, pass_thres, 
                                          knn, query_key=query_key, db_key=db_key)
    print('>> Total elapsed time for evaluation: {:.2f}'.format(time.time()-start))
    lprint('Test Avg percent: {}, avg similairty {}'.format(avg_percent, avg_sim), log)
    
