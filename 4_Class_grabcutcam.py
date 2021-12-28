from torchvision import transforms
import torch
import numpy as np
import timeit
import gc
import pandas as pd
from pathlib import Path

import utils.image as im
import utils.cam as cam
import utils.grabcut as gcut
import utils.metrics as m
import utils.json as json
import utils.path as path
from utils.VOCSegmentation import VOCSegmentation


# PATHS

root_path   = path.goback_from_current_dir(0)
json_path   = root_path + 'json\\'
output_path = root_path + 'output\\'

# SETUP

camnet = cam.Cam()

data = VOCSegmentation(root = root_path,
                           year = '2012',
                           image_set = 'trainval',
                           download = False,
                           transform = transforms.ToTensor(),
                           target_transform = transforms.ToTensor(),
                           transforms = None,
                           target = 'Class')
data = iter(torch.utils.data.DataLoader(data,
                                        batch_size = 1,
                                        shuffle = False,
                                        num_workers = 0))

annotations = json.open_json(json_path + "voc-class-annotations")
N = len(annotations)
annotations = iter(annotations.items())

classes    = json.open_json(json_path + "voc-classes")
thresholds = (np.arange(1, 10) / 10).tolist()
metrics    = ['TP', 'FN', 'FP']

time = []

measures = np.zeros((len(thresholds), len(classes), len(metrics)), dtype=int)

# MEASUREMENT

for i in range(N):
    gc.collect()
    start = timeit.default_timer()
    
    img, sgm = next(data)
    img, sgm = torch.squeeze(img), torch.squeeze(sgm)

    img_pil  = img
    img_cv2  = im.pil_to_cv2(img.numpy())
    sgm      = im.f1_to_f255(sgm.numpy())
    undef    = np.where(sgm == 255, 1, 0).astype(bool)
    
    name, annots = next(annotations)

    for j, t in enumerate(thresholds):
        for c in annots:
            k = classes.index(c)

            img_cam    = camnet.get_top(img_pil, c)
            img_cam    = cam.cam_process(img_pil, img_cam)
            gcmask_cam = cam.cam_to_gcmask(img_cam, t) 
            _, _, pred = gcut.grabcut(img_cv2, gcmask_cam, mode = 'MASK')
            pred       = pred.astype(bool) & ~ undef
            
            true    = m.true_mask(sgm, k + 1)
        
            measures[j][k] = np.add(measures[j][k], m.TP_FN_FP(true, pred))
        
    stop = timeit.default_timer()
    time.append(stop - start)
    if i % 20 == 0:
        print(f'Image nb {i}')
        print(f'Time spent               = ' + m.time_str(np.sum(time)))
        print(f'Estimated time remaining = ' + m.time_str(np.mean(time) * (N - 1 - i)))
        print()

# DATAFRAME CREATION

index = pd.MultiIndex.from_product([thresholds, classes, range(len(metrics))], names = ['0', '1', '2'])
df = pd.DataFrame({'A': measures.flatten()}, index=index)['A']
df = df.unstack(level='2')
df.columns = metrics
df.index.names = ['Threshold', 'Class']

# SAVING OUTPUT

Path(output_path).mkdir(parents = True, exist_ok = True)
df.to_csv(output_path + 'class_grabcutcam.csv')