# Arguments used for networks
class Args_voc(object):
    def __init__(self):
        self.backbone = 'resnet'
        self.out_stride = 16
        self.dataset = 'pascal'
        self.use_sbd = False
        self.workers = 4
        self.base_size = (513, 513)
        self.crop_size = (513, 513)
        self.scale_ratio = (0.5, 2.0)  # random scale from 0.5 to 2.0
        self.sync_bn = None
        self.freeze_bn = False
        self.loss_type = 'ce'
        self.epochs = None
        self.start_epoch = 0
        self.batch_size = None
        self.test_batch_size = None
        self.use_balanced_weights = False
        self.lr = None
        self.lr_scheduler = 'poly'
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.nesterov = False
        self.no_cuda = False
        self.gpu_ids = '0'
        self.seed = 1
        self.resume = None
        self.checkname = None
        self.ft = False
        self.eval_interval = 1
        self.no_val = False


class Args_occ5000(object):
    def __init__(self):
        self.backbone = 'resnet'
        self.out_stride = 16
        self.dataset = 'occ5000'
        self.use_sbd = False
        self.workers = 4
        self.base_size = (1361, 305)  # scale on base_size from 0.5 to 2.0, should set to be same as image size
        self.crop_size = (1361, 305)  # [h_crop, w_crop], crop_size = k * output_stride + 1, make crop_size as large as you can
        self.scale_ratio = (0.5, 2.0)  # random scale from 0.5 to 2.0
        self.sync_bn = None
        self.freeze_bn = False
        self.loss_type = 'ce'
        self.test_batch_size = None
        self.use_balanced_weights = False
        self.lr = None
        self.lr_scheduler = 'poly'
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.nesterov = False
        self.no_cuda = False
        self.seed = 1
        self.resume = None  # '/home/kidd/kidd1/pytorch-deeplab-xception/run/occ5000/deeplab_kinematic/2020-01-19-01:06:06/35-0.497667546414435.pth.tar' # path to resume model file
        self.ft = False
        self.eval_interval = 1  # eval on eval set interval
        self.no_val = False
        # commonly used parameters
        self.use_kinematic = True
        self.no_flip = True  # flip images
        self.epochs = 50
        self.checkname = 'deeplab_v3+_noflip_kinematic'
        self.gpu_ids = '0'
        self.start_epoch = 0
        self.batch_size = 2
