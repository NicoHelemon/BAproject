from torchvision import transforms
import torch
import numpy as np
import timeit
import pandas as pd
import gc
from pathlib import Path

import utils.image as im
import utils.cam as cam
import utils.grabcut as gcut
import utils.voc as voc
import utils.json as json
import utils.path as path
from utils.VOCSegmentation import VOCSegmentation

FB_threshold = 0.2

root_path = path.goback_from_current_dir(0)
json_path = root_path + 'json\\'
output_path  = root_path + r'sessions\4 Class grabcut-cam\threshold_0' + str(int(FB_threshold * 10)) + '\\'

Path(output_path).mkdir(parents=True, exist_ok=True)

camnet = cam.Cam()

data_tbl = VOCSegmentation(root = root_path,
                           year = '2012',
                           image_set = 'trainval',
                           download = False,
                           transform = transforms.ToTensor(),
                           target_transform = transforms.ToTensor(),
                           transforms = None,
                           target = 'Class')

data = iter(torch.utils.data.DataLoader(data_tbl,
                                        batch_size = 1,
                                        shuffle = False,
                                        num_workers = 0))

annotations = json.open_json(json_path + "voc-class-annotations")
N = len(annotations)
annotations = iter(annotations.items())

classes = json.open_json(json_path + "voc-classes")

IoU  = []
Acc  = []
time = []

for i in range(N):
    gc.collect()
    start = timeit.default_timer()
    
    img, sgm = next(data)
    img, sgm = torch.squeeze(img), torch.squeeze(sgm)

    img_pil = img
    img_cv2 = im.pil_to_cv2(img.numpy())
    sgm     = im.f1_to_f255(sgm.numpy())
    
    name, annots = next(annotations)
    
    for c in annots:
        
        img_cam = camnet.get_top(img_pil, c)
        img_cam = cam.cam_process(img_pil, img_cam)
        gcmask_cam = cam.cam_to_gcmask(img_cam, a = 0.0, b = FB_threshold, c = 1.0)
        _, _, pred = gcut.grabcut(img_cv2, gcmask_cam, mode = 'MASK')
        pred = pred.astype(bool)
        
        true = voc.true_mask(sgm, classes.index(c) + 1).astype(bool)
        
        IoU.append(voc.IoU(true, pred, sgm))
        Acc.append(voc.accuracy(true, pred, sgm))
        
    stop = timeit.default_timer()
    time.append(stop - start)
    if i % 20 == 0:
        print(f'Image nb {i}')
        print(f'Time spent               = ' + voc.time_str(np.sum(time)))
        print(f'Estimated time remaining = ' + voc.time_str(np.mean(time) * (N - 1 - i)))
        print(f'Mean IoU                 =  {np.mean(IoU) :.3f}')
        print(f'Mean Acc                 =  {np.mean(Acc) :.3f}')
        print()

print(f'Mean IoU = {np.mean(IoU):.3f}')
print(f'Mean Acc = {np.mean(Acc):.3f}')

df = pd.read_csv(json_path + 'voc-class-annotations.csv')

df['IoU'] = IoU
df['Accuracy'] = Acc

df.to_csv(output_path + 'IoU.csv', index = False)