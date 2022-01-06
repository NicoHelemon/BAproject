from torchvision import transforms
import torch
import numpy as np
import timeit
import gc
import pandas as pd
from pathlib import Path

import utils.image as im
import utils.cam as cam
import utils.metrics as m
import utils.json as json
import utils.path as path
import utils.segmentation as sg
from utils.VOCSegmentation import VOCSegmentation


# PATHS

root_path   = path.goback_from_current_dir(0)
json_path   = root_path + 'json\\'
output_path = root_path + 'output\\'

Path(output_path).mkdir(parents = True, exist_ok = True)

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
metrics    = ['TP', 'FN', 'FP', 'TN']
techniques = ['cam', 'grabcutcam']

measures = np.zeros((len(techniques), len(thresholds), len(classes), len(metrics)), dtype=int)

time = []

# MEASUREMENT

for i in range(N):
    gc.collect()
    start = timeit.default_timer()
    
    img, true_sgms = next(data)
    img, true_sgms = torch.squeeze(img), torch.squeeze(true_sgms)

    img_pil   = img
    img_cv2   = im.pil_to_cv2(img.numpy())
    true_sgms = im.f1_to_f255(true_sgms.numpy())
    undef     = np.where(true_sgms == 255, 1, 0).astype(bool)
    
    name, annots = next(annotations)

    for c in annots:
        k = classes.index(c)
        true_sgm = m.true_mask(true_sgms, k + 1)

        _, _, img_cam = camnet.get_top_voc_to_imagenet(img_pil, c)

        for j, t in enumerate(thresholds):
        
            pred0 = sg.sgm_cam(img_cam, t)
            pred1 = sg.sgm_grabcut_cam(img_cv2, img_cam, t)
        
            measures[0][j][k] = np.add(measures[0][j][k], m.TP_FN_FP_TN(true_sgm, pred0, undef))
            measures[1][j][k] = np.add(measures[1][j][k], m.TP_FN_FP_TN(true_sgm, pred1, undef))
        
    stop = timeit.default_timer()
    time.append(stop - start)
    if i % 20 == 0:
        print(f'Image nb {i}')
        print(f'Time spent               = ' + m.time_str(np.sum(time)))
        print(f'Estimated time remaining = ' + m.time_str(np.mean(time) * (N - 1 - i)))
        print()

# DATAFRAME CREATION

def df_creation_saving(array, thresholds, name):
    index = pd.MultiIndex.from_product([thresholds, classes, range(len(metrics))], names = ['0', '1', '2'])
    df = pd.DataFrame({'A': array.flatten()}, index = index)['A']
    df = df.unstack(level = '2')
    df.columns = metrics
    df.index.names = ['Threshold', 'Class']

    df.to_csv(output_path + f'class_{name}.csv')

# SAVING OUTPUT

df_creation_saving(measures[0], thresholds, 'cam')
df_creation_saving(measures[1], thresholds, 'grabcutcam')