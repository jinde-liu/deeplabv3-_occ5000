import os
import shutil
import torch
from collections import OrderedDict
import glob
import time

class Saver(object):
    # make save dir in the time format
    def __init__(self, args):
        self.args = args
        self.directory = os.path.join('run', args.dataset, args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, '*')))
        run_id = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

        self.experiment_dir = os.path.join(self.directory, run_id)
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk and copy it to ./run dir if it is the best
        state: state dict
        is_best: if current checkpoint is best copy it to ./run
        filename: checkpoint file name
        """
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            epoch = state['epoch']
            class_miou = state['class_miou']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred) + '\n')
                f.write('background: {:.2%}\n'.format(class_miou[0]))
                f.write('hair: {:.2%}\n'.format(class_miou[1]))
                f.write('face: {:.2%}\n'.format(class_miou[2]))
                f.write('torso: {:.2%}\n'.format(class_miou[3]))
                f.write('left_arm: {:.2%}\n'.format(class_miou[4]))
                f.write('right_arm: {:.2%}\n'.format(class_miou[5]))
                f.write('left_hand: {:.2%}\n'.format(class_miou[6]))
                f.write('right_hand: {:.2%}\n'.format(class_miou[7]))
                f.write('left_leg: {:.2%}\n'.format(class_miou[8]))
                f.write('right_leg: {:.2%}\n'.format(class_miou[9]))
                f.write('left_foot: {:.2%}\n'.format(class_miou[10]))
                f.write('right_foot: {:.2%}\n'.format(class_miou[11]))
                f.write('accessory: {:.2%}\n'.format(class_miou[12]))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    path = os.path.join(run, 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, str(epoch)+'-'+str(best_pred)+'-model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, str(epoch)+'-'+str(best_pred)+'-model_best.pth.tar'))

    # save parameters in 'parameter.txt' --kidd
    def save_experiment_config(self):
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        p = OrderedDict()
        p['datset'] = self.args.dataset
        p['backbone'] = self.args.backbone
        p['out_stride'] = self.args.out_stride
        p['lr'] = self.args.lr
        p['lr_scheduler'] = self.args.lr_scheduler
        p['loss_type'] = self.args.loss_type
        p['epoch'] = self.args.epochs
        p['base_size'] = self.args.base_size
        p['crop_size'] = self.args.crop_size

        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()