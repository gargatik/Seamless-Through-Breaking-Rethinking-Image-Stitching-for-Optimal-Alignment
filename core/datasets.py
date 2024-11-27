import numpy as np
import torch
import torch.utils.data as data

import os
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse

        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):
        #print(self.flow_list[index])
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                #print(worker_info.id)
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        
        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        return img1, img2, flow, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)

class MpiSintel_submission(FlowDataset):
    def __init__(self, aug_params=None, split='test', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel_submission, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))


        

class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean'):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        if split == 'test':
            self.is_test = True

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))



class FlyingChairs(FlowDataset):
    def __init__(self, aug_params=None, split='train', root='datasets/FlyingChairs_release/data'):
        super(FlyingChairs, self).__init__(aug_params)

        images = sorted(glob(osp.join(root, '*.ppm')))
        flows = sorted(glob(osp.join(root, '*.flo')))
        assert (len(images)//2 == len(flows))

        split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
        for i in range(len(flows)):
            xid = split_list[i]
            if (split=='training' and xid==1) or (split=='validation' and xid==2):
                self.flow_list += [ flows[i] ]
                self.image_list += [ [images[2*i], images[2*i+1]] ]


class FlyingThings3D(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/FlyingThings3D', dstype='frames_cleanpass', split='training'):
        super(FlyingThings3D, self).__init__(aug_params)

        split_dir = 'TRAIN' if split == 'training' else 'TEST'
        for cam in ['left']:
            for direction in ['into_future', 'into_past']:
                image_dirs = sorted(glob(osp.join(root, dstype, f'{split_dir}/*/*')))
                image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

                flow_dirs = sorted(glob(osp.join(root, f'optical_flow/{split_dir}/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')) )
                    flows = sorted(glob(osp.join(fdir, '*.pfm')) )
                    for i in range(len(flows)-1):
                        if direction == 'into_future':
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[i] ]
                        elif direction == 'into_past':
                            self.image_list += [ [images[i+1], images[i]] ]
                            self.flow_list += [ flows[i+1] ]
class FlyingThings3DSubset(FlowDataset):
    def __init__(self, aug_params=None, root='/workspace/data/things_dataset/subset/FlyingThings3D_subset', dstype='image_clean', split='training'):
        super(FlyingThings3DSubset, self).__init__(aug_params)

        split_dir = 'train' if split == 'training' else 'val'
        for cam in ['left']:
            # for direction in ['into_future', 'into_past']:
            for direction in ['into_future']:
                # root/ split_dir / dstype / cam
                idir = osp.join(root, split_dir, dstype, cam)
                fdir = osp.join(root, split_dir, 'flow', cam, direction)

                images = sorted(glob(osp.join(idir, '*.png')) )# 21818
                flows = sorted(glob(osp.join(fdir, '*.flo')) ) # 19640
                
                for j in range(len(flows)-1):
                    flow_name = flows[j].split('/')[-1].split('.')[0]
                    i = int(flow_name)
                    if direction == 'into_future':
                        # check the name is the same in images and flows
                        if images[i].split('/')[-1].split('.')[0] == flows[j].split('/')[-1].split('.')[0]:
                            self.image_list += [ [images[i], images[i+1]] ]
                            self.flow_list += [ flows[j] ]


      

class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='datasets/KITTI'):
        super(KITTI, self).__init__(aug_params, sparse=True)
        if split == 'testing':
            self.is_test = True

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if split == 'training':
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


class HD1K(FlowDataset):
    def __init__(self, aug_params=None, root='datasets/HD1k'):
        super(HD1K, self).__init__(aug_params, sparse=True)

        seq_ix = 0
        while 1:
            flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flows) == 0:
                break

            for i in range(len(flows)-1):
                self.flow_list += [flows[i]]
                self.image_list += [ [images[i], images[i+1]] ]

            seq_ix += 1



class CADataset(FlowDataset):
    def __init__(self, data_dir, aug_params=None, phase='train'):
        super(CADataset, self).__init__(aug_params, sparse=False)

        assert phase in ['train', 'val', 'test']        
        
        if phase == 'test':
            self.is_test = True
        else:
            print(f"phase : {phase}, but CADataset no flow ground truth , flow is zero")

        self.base_path = data_dir
        self.list_path = self.base_path + '/{}.txt'.format(phase)
        self.data_infor = open(self.list_path, 'r').readlines()
        for index in range(len(self.data_infor)):
            img_names = self.data_infor[index].replace('\n', '')
            img_names = img_names.split(' ')
            img1_path = self.base_path + 'img/' + img_names[0]
            img2_path = self.base_path + 'img/' + img_names[1]

            self.image_list += [ [img1_path, img2_path] ]
            self.extra_info += [ (img_names[0], img_names[1]) ] 
        
        
    def __getitem__(self, index):
        #print(self.flow_list[index])
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                #print(worker_info.id)
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        
        # flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)



        flow = np.zeros((img1.shape[0], img1.shape[1], 2), dtype=np.float32)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.zeros(2, img1.shape[1], img1.shape[2]).float()


        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        return img1, img2, flow, valid.float()
    


from collections import OrderedDict
class UDISDataset(FlowDataset):
    def __init__(self, data_dir, aug_params=None, phase='training', return_depth=False, clip_num_data=None):
        super(UDISDataset, self).__init__(aug_params, sparse=False)

        assert phase in ['training','testing']      

        self.return_depth = return_depth
        
        if phase == 'testing':
            self.is_test = True
        else:
            print(f"phase : {phase}, but UDISataset no flow ground truth , flow is zero")

        self.data_path = data_dir + phase + '/'
        self.datas = OrderedDict()
        
        datas =  glob(os.path.join(self.data_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input1' or data_name == 'input2' :
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
            if return_depth and data_name == 'depth2':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
        print(self.datas.keys())

        total_len = min( len(self.datas['input1']['image']), len(self.datas['input2']['image']))
        
        for index in range(total_len):
            
            img1_path = self.datas['input1']['image'][index]
            img2_path = self.datas['input2']['image'][index]
            if return_depth:
                depth2_path = self.datas['depth2']['image'][index]
                self.image_list += [ [img1_path, img2_path, depth2_path] ]
                self.extra_info += [ (img1_path, img2_path, depth2_path) ] 
            else:
                self.image_list += [ [img1_path, img2_path] ]
                self.extra_info += [ (img1_path, img2_path) ] 

        if clip_num_data!=None:
            self.image_list = self.image_list[:clip_num_data]
            self.extra_info = self.extra_info[:clip_num_data] 
            print(f"clip_num_data : {clip_num_data}, (origin len:{total_len})")
        
        
    def __getitem__(self, index):
        #print(self.flow_list[index])
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                #print(worker_info.id)
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
     
        
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        flow = np.zeros((img1.shape[0], img1.shape[1], 2), dtype=np.float32)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.return_depth:
                raise NotImplementedError("UDISDataset no support return_depth on augmentor")

            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.zeros(2, img1.shape[1], img1.shape[2]).float()


        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        if self.return_depth:
            depth2 = frame_utils.read_gen(self.image_list[index][2])
            depth2 = np.array(depth2).astype(np.uint8)
            if len(depth2.shape) == 2:
                depth2 = np.tile(depth2[...,None], (1, 1, 3))
            else:
                depth2 = depth2[..., :3]
            depth2 = torch.from_numpy(depth2).permute(2, 0, 1).float()
            depth2 = depth2.mean(dim=0, keepdim=True)
            return img1, img2, depth2, valid.float()
        else:
            return img1, img2, flow, valid.float()



def fetch_dataloader(args, TRAIN_DS='C+T+K+S+H'):
    """ Create the data loader for the corresponding trainign set """


    if args.stage == 'chairs':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = FlyingChairs(aug_params, split='training')
    
    elif args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass')
        train_dataset = clean_dataset + final_dataset
    elif args.stage == 'things_subset':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
        train_dataset = FlyingThings3DSubset(aug_params, dstype='image_clean')

    elif args.stage == 'sintel':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
        sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
        sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

        if TRAIN_DS == 'C+T+K+S+H':
            kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
            hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
            train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti  + 5*hd1k + things

        elif TRAIN_DS == 'C+T+K/S':
            train_dataset = 100*sintel_clean + 100*sintel_final + things

    elif args.stage == 'kitti':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
        train_dataset = KITTI(aug_params, split='training')
    elif args.stage == 'dtu':
        from core.dtu_dataset import DTU
        train_dataset = DTU( params=args, aug_params=None, sparse=True, phase="train")
    elif 'ca' in args.stage :
        aug_params = {'crop_size': args.image_size, 'min_scale': 1.0, 'max_scale': 1.0, 'do_flip': False}
        train_dataset = CADataset(data_dir=args.ca_dataset.data_dir, aug_params=aug_params, phase="train")
    elif 'udis' in args.stage :
        clip_num_data = None if not hasattr(args.udis_dataset, 'clip_num_data') else args.udis_dataset.clip_num_data
        train_dataset = UDISDataset(data_dir=args.udis_dataset.data_dir, aug_params=None, phase="training", return_depth=args.udis_dataset.return_depth, clip_num_data=clip_num_data)



    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=8, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader

if __name__ == "__main__":
    data = FlyingThings3DSubset(aug_params=None, root='/data4/things_dataset/subset/FlyingThings3D_subset', dstype='image_clean', split='training')
    for img, flo in zip(data.image_list, data.flow_list):
        print(img, flo)
        break
