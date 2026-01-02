# Display joints and vertices estimated by wilor.
# conda activate sign
# cd exter/manopth

import numpy as np
import torch
from pathlib import Path

from manopth.manolayer import ManoLayer
from manopth import demo_cw

batch_size = 1
ncomps = 6
nfull_comps = ncomps + 3  # Add global orientation dims to PCA
random_pcapose = torch.rand(batch_size, nfull_comps)
mano_root = 'manopth_mano/models'
mano_layer = ManoLayer(mano_root=mano_root)
#verts, joints = mano_layer(random_pcapose)

data_dir = Path.home(
) / '/home/colin/Projects/SignTracking/signtrace/extern/wilor/chicago_out/'
frame_id = '0005'
hand = ['right', 'left'][0]
joints_file = data_dir / f'{frame_id}_joints_{hand}.txt'
verts_file = data_dir / f'{frame_id}_verts_{hand}.txt'

joints = np.loadtxt(joints_file, delimiter=',')
verts = np.loadtxt(verts_file, delimiter=',')

joints = torch.from_numpy(joints).unsqueeze(0)
verts = torch.from_numpy(verts).unsqueeze(0)

hand_info = { \
    'joints': joints,
    'verts': verts,
             }
#    'faces': mano_layer.th_faces}

demo_cw.display_hand(hand_info,
                     mano_faces=mano_layer.th_faces,
                     ax=None,
                     alpha=0.2,
                     batch_idx=0,
                     show=True)
