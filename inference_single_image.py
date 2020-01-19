from PIL import Image
import numpy as np
import torch
import torchvision.transforms as tr
from modeling.deeplab import DeepLab
from dataloaders.utils import decode_segmap
from utils.metrics import Evaluator
import matplotlib.pyplot as plt
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load("/home/kidd/kidd1/pytorch-deeplab-xception/run/occ5000/deeplab-resnet/no_spl_model_best.pth.tar")

model = DeepLab(num_classes=13,
                backbone='resnet',
                output_stride=16,
                sync_bn=True,
                freeze_bn=False)

model.load_state_dict(checkpoint['state_dict'])
model.eval()
model.to(device)
torch.set_grad_enabled(False)
def transform(image):
    return tr.Compose([
        tr.ToTensor(),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])(image)

# Read eval image and gt
image = Image.open('/home/kidd/kidd1/Occ5000/Occ4000/images/CAM31-2014-03-18-20140318120853-20140318121449-bak_floor4-\
cam1031000-20140318120853-tarid82-frame3165-line1-pos744-12-822-266-box723-6-108-255.png')
gt_im = Image.open('/home/kidd/kidd1/Occ5000/Occ4000/annotationsLast/CAM31-2014-03-18-20140318120853-20140318121449-bak\
_floor4-cam1031000-20140318120853-tarid82-frame3165-line1-pos744-12-822-266-box723-6-108-255.png')
gt = np.array(gt_im)
gt_rgb = decode_segmap(gt, dataset="occ5000")

# Inference and set the visual color map
inputs = transform(image).to(device)
output = model(inputs.unsqueeze(0)).squeeze().cpu().numpy()
pred = np.argmax(output, axis=0)
pred_rgb = decode_segmap(pred, dataset="occ5000")

plt.subplot(1,3,1)
plt.imshow(image)
plt.subplot(1,3,2)
plt.imshow(gt_rgb)
plt.subplot(1,3,3)
plt.imshow(pred_rgb)
plt.show()

eval = Evaluator(13)
eval.reset()
eval.add_batch(gt, pred)
miou = eval.Mean_Intersection_over_Union()
print(miou)
class_miou = eval.Class_Intersection_over_Union()
print(class_miou)





