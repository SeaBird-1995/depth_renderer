'''
This is a vtk library to visualize point cloud
from multiple depth maps aligned with camera settings.

Author: ynie
Date: Jan 2020
'''
import sys
sys.path.append('./')
from data_config import shapenet_rendering_path, total_view_nums
import os
from pc_painter import PC_from_DEP
from data_config import camera_setting_path
import numpy as np


if __name__ == '__main__':
    depth_sample_dir = '02818832/e91c2df09de0d4b1ed4d676215f46734'
    n_views = 20
    assert n_views <= total_view_nums # there are total 20 views surrounding an object.
    view_ids = range(1, n_views+1)
    metadata_dir = os.path.join(shapenet_rendering_path, depth_sample_dir)
    pc_from_dep = PC_from_DEP(metadata_dir, camera_setting_path, view_ids, with_normal=True)

    point_cloud = pc_from_dep.point_clouds['pc'][14]
    print(point_cloud.shape, point_cloud.min(), point_cloud.max())
    np.savetxt("point_cloud_14.xyz", point_cloud)
    # pc_from_dep.draw_depth(view='all')
    # pc_from_dep.draw_color(view='all')
    # pc_from_dep.draw3D()