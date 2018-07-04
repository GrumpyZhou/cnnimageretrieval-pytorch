import os
import numpy as np
import torch 
from torch.autograd import Variable
import torch.utils.data as data
import time

from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.datasets.datahelpers import collate_tuples, default_loader, imresize

class CustomizeTuplesDataset(data.Dataset):
    def __init__(self, name, mode, db_file, ims_root,
                 imsize=362, snum=3, dnum=2, qsize=2000,
                 transform=None, loader=default_loader):

        # Load db file
        self.mode = mode
        self.name = name
        self.ims_root = ims_root
        self.imsize = imsize
        self.snum = snum
        self.dnum = dnum
        self.transform = transform
        self.loader = loader
        db = np.load(db_file).item()[mode]
        self.qkey = db['qkey']
        self.dbkey = db['dbkey']
        self.qims = [os.path.join(ims_root, db['datasets'][db['qcids'][i]], db['qims'][i]) for i in range(len(db['qcids']))]
        self.dbims = [os.path.join(ims_root, db['datasets'][db['dbcids'][i]], db['dbims'][i]) for i in range(len(db['dbcids']))]
        self.clusters = db['datasets']
        self.qcids = db['qcids']
        self.dbcids = db['dbcids']
        self.qpool = db['qids']
        self.ppool = db['pids']
        self.snn = db['snn']
        self.qsize = min(qsize, len(self.qpool))
        
    def create_epoch_tuples(self, net):
        print('>> Creating tuples for an epoch of {}-{}...'.format('7Scene', self.mode))
    
        # Draw qsize random queries for tuples
        idxs2qpool = torch.randperm(len(self.qpool))[:self.qsize]
        self.qidxs = [self.qpool[i] for i in idxs2qpool]
        self.pidxs = [self.ppool[i] for i in idxs2qpool]

        # Create dummy nidxs useful when only positives used for training
        if (self.snum + self.dnum) == 0:
            self.nidxs = [[] for _ in range(len(self.qidxs))]
            return
        
        # Extract descriptors for query images
        net.cuda()
        net.eval()
        t1 = time.time()  
        print('>> Extracting descriptors for all query images...')
        loader = torch.utils.data.DataLoader(
            ImagesFromList(root='', images=self.qims, imsize=self.imsize, transform=self.transform),
            batch_size=1, shuffle=False, num_workers=8, pin_memory=True
        )
        qimvecs = torch.Tensor(net.meta['outputdim'], len(self.qims)).cuda()
        for i, input in enumerate(loader):
            print('\r>>>> {}/{} done...'.format(i+1, len(self.qims)), end='')
            qimvecs[:, i] = net(Variable(input.cuda())).data.squeeze()
        print('Total time from extracting query image descriptors: {}s, scriptor size {}'.format(time.time()-t1, qimvecs.size()))

        # Extract descriptors for db images
        print('>> Extracting descriptors for all db images...')
        if self.qkey == self.dbkey:  # Query images are the same as db images
            dbimvecs = qimvecs
        else:
            t1 = time.time()  
            loader = torch.utils.data.DataLoader(
                ImagesFromList(root='', images=self.dbims, imsize=self.imsize, transform=self.transform),
                batch_size=1, shuffle=False, num_workers=8, pin_memory=True
            )
            dbimvecs = torch.Tensor(net.meta['outputdim'], len(self.dbims)).cuda()
            for i, input in enumerate(loader):
                print('\r>>>> {}/{} done...'.format(i+1, len(self.dbims)), end='')
                dbimvecs[:, i] = net(Variable(input.cuda())).data.squeeze()
            print('Total time from extracting db image descriptors: {}s, scriptor size {}'.format(time.time()-t1, dbimvecs.size()))

        
        # Search for negative pairs
        t1 = time.time()  
        self.nidxs = []
        avg_ndist = torch.Tensor([0]).cuda()
        n_ndist = torch.Tensor([0]).cuda()
        for qid in self.qidxs:
            qvecs = qimvecs[:, qid]
            qcid = self.qcids[qid]

            # Pick snum hard negative images from the same cluster
            nnpool = self.snn[qid]
            snvecs = dbimvecs[:, nnpool]
            scores = torch.mv(snvecs.t(), qvecs)
            scores, ranks = torch.sort(scores, dim=0, descending=True)
            snid = []
            r = 0
            while len(snid) < self.snum:
                # Ignore the self frame
                psnid = nnpool[ranks[r]]
                r += 1
                if self.qkey == self.dbkey and psnid == qid:
                    continue
                snid.append(psnid)
                avg_ndist += torch.pow(qvecs - dbimvecs[:, psnid] + 1e-6, 2).sum(dim=0).sqrt()
                n_ndist += 1

            # Pick any dnum negatives from a different cluster
            dnid = []
            dcid = []
            while len(dnid) < self.dnum:
                pdnid = int(torch.randint(low=0, high=len(self.dbims), size=(1,)))
                pdcid = self.dbcids[pdnid] 
                if pdcid == qcid or pdcid in dcid:  # A different cluster from the query and selected negative ims
                    continue
                dnid.append(pdnid)
                dcid.append(pdcid)
                avg_ndist += torch.pow(qvecs - dbimvecs[:, pdnid] + 1e-6, 2).sum(dim=0).sqrt()
                n_ndist += 1
            self.nidxs.append(snid + dnid)
        print('>>>> Average negative distance: {:.2f}'.format(list((avg_ndist/n_ndist).cpu())[0]))
        print('Total search time {}s'.format(time.time() - t1))
        print('>>>> Dataset Preparation Done')
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            images tuple (q,p,n1,...,nN): Loaded train/val tuple at index of self.qidxs
        """
        if self.__len__() == 0:
            raise(RuntimeError("List qidxs is empty. Run ``dataset.create_epoch_tuples(net)`` method to create subset for train/val!"))

        output = []
        output.append(self.loader(self.qims[self.qidxs[index]])) # query image
        output.append(self.loader(self.dbims[self.pidxs[index]])) # positive image
        
        # negative images
        for i in range(len(self.nidxs[index])):
            output.append(self.loader(self.dbims[self.nidxs[index][i]]))
        if self.imsize is not None:
            output = [imresize(img, self.imsize) for img in output]
        if self.transform is not None:
            output = [self.transform(output[i]).unsqueeze_(0) for i in range(len(output))]
        target = torch.Tensor([-1, 1] + [0]*len(self.nidxs[index]))

        return output, target

    def __len__(self):
        if not self.qidxs:
            return 0
        return len(self.qidxs)
    
    def __repr__(self):
        fmt_str = self.__class__.__name__ + '\n'
        fmt_str += '    Name and mode: {} {}\n'.format(self.name, self.mode)
        fmt_str += '    Number of query images: {}\n'.format(len(self.qims))
        fmt_str += '    Number of db images: {}\n'.format(len(self.dbims))
        fmt_str += '    Number of training tuples: {}\n'.format(len(self.qpool))
        fmt_str += '    Number of same cluster negatives per tuple: {}\n'.format(self.snum)
        fmt_str += '    Number of different cluster negatives per tuple: {}\n'.format(self.dnum)
        fmt_str += '    Number of tuples processed in an epoch: {}\n'.format(self.qsize)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str 
