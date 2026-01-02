# Modified version of manopth/demo.py
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch

from manopth.manolayer import ManoLayer

# MANO landmarks and connections.
# joint ids from: Li, Guiqing, et al. "3D hand reconstruction from a single image based on biomechanical constraints." The Visual Computer 37 (2021): 2699-2711. (surely exists elsewhere?)
# todo: label joints
mano_landmarks = {
    'palm': [0],
    'thumb': [1, 2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20],
}

mano_joint_connections = [ \
    mano_landmarks['palm'] + mano_landmarks[finger] for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']]
#mano_joint_connections += \
#    [[5, 9, 13, 17]]  # brass knuckles, baby

mycmap = plt.colormaps['viridis']
mycolors = mycmap(np.linspace(0, 1, len(mano_landmarks) + 1))

# def generate_random_hand(batch_size=1, ncomps=6, mano_root='mano/models'):
#     nfull_comps = ncomps + 3  # Add global orientation dims to PCA
#     random_pcapose = torch.rand(batch_size, nfull_comps)
#     mano_layer = ManoLayer(mano_root=mano_root)
#     verts, joints = mano_layer(random_pcapose)
#     return {'verts': verts, 'joints': joints, 'faces': mano_layer.th_faces}


def display_hand(hand_info,
                 mano_faces=None,
                 ax=None,
                 alpha=0.2,
                 batch_idx=0,
                 show=True):
    """
    Displays hand batch_idx in batch of hand_info, hand_info as returned by
    generate_random_hand
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    verts = hand_info['verts'][batch_idx]
    joints = hand_info['joints'][batch_idx]
    if mano_faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.1)
    else:
        mesh = Poly3DCollection(verts[mano_faces], alpha=alpha)
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')
    for j, joint_connection in enumerate(mano_joint_connections):
        ax.plot(joints[joint_connection, 0],
                joints[joint_connection, 1],
                joints[joint_connection, 2],
                color=mycolors[j],
                lw=4)  # CW
    cam_equal_aspect_3d(ax, verts.numpy())
    if show:
        plt.show()


def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)
