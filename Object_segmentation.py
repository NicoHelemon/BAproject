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
import utils.segmentation as sgm
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
                           target = 'Object')
data = iter(torch.utils.data.DataLoader(data,
                                        batch_size = 1,
                                        shuffle = False,
                                        num_workers = 0))

annotations = json.open_json(json_path + "voc-object-annotations-clean")
N = len(annotations)
annotations = iter(annotations.items())

classes    = json.open_json(json_path + "voc-classes")
thresholds = (np.arange(1, 10) / 10).tolist()
metrics    = ['TP', 'FN', 'FP', 'TN']
techniques = ['grabcut', 'grabcutcam_PF_PB', 'grabcutcam_F_PF', 'grabcutcam_F_PB']

measures = np.zeros((len(techniques), len(thresholds), len(classes), len(metrics)), dtype=int)

time = []

# MEASUREMENT

for i in range(N):
    gc.collect()
    start = timeit.default_timer()
    
    img, true_sgms, undef = im.process(*next(data))
    _, annots = next(annotations)

    for l, annot in enumerate(annots):
        true_sgm = m.true_mask(true_sgms, l + 1)

        c, _, cbbox = annot
        k = classes.index(c)

        #pred0 = sgm.sgm_grabcut(img, cbbox)
        #measures[0][0][k] = np.add(measures[0][0][k], m.TP_FN_FP_TN(true_sgm, pred0, undef))

        _, _, img_cam = camnet.get_top_voc_to_imagenet(img, c)
        img_cam = im.bitwise_and(img_cam, im.cbbox_mask(img_cam.shape[:2], cbbox))

        for j, t in enumerate(thresholds):

            #pred1 = sgm.sgm_grabcut_cam(img, img_cam, t, mode = 'PF_PB', cbbox = cbbox)
            pred2 = sgm.sgm_grabcut_cam(img, img_cam, t, mode = 'F_PF', cbbox = cbbox)
            pred3 = sgm.sgm_grabcut_cam(img, img_cam, t, mode = 'F_PB', cbbox = cbbox)
        
            #measures[1][j][k] = np.add(measures[1][j][k], m.TP_FN_FP_TN(true_sgm, pred1, undef))
            measures[2][j][k] = np.add(measures[2][j][k], m.TP_FN_FP_TN(true_sgm, pred2, undef))
            measures[3][j][k] = np.add(measures[3][j][k], m.TP_FN_FP_TN(true_sgm, pred3, undef))
        
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

    df.to_csv(output_path + f'object_{name}.csv')

# SAVING OUTPUT

#df_creation_saving(measures[0][0, :], ['-'], 'grabcut')
#df_creation_saving(measures[1], thresholds, 'grabcutcam_PF_PB')
df_creation_saving(measures[2], thresholds, 'grabcutcam_F_PF')
df_creation_saving(measures[3], thresholds, 'grabcutcam_F_PB')

