class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'occ5000':
            return '/home/kidd/kidd1/Occ5000'
        elif dataset == 'pascal':
            return '/home/kidd/kidd1/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
