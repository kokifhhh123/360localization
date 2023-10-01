import numpy as np
import torch
import math
import open3d as o3d
def to_lnglat(xy, w, h):
    '''
    xy = np.array([(0, 0), (1, 1)])
    '''

    x = xy[:,0]
    y = xy[:,1]
    lng = x / (w / 360) - 180
    #lng = x/R
    #lat = -y / (h / 180) + 90
    lat = y / (h / 180)
    lng = 360 - ((lng + 360) % 360)
    #lat = lat - 90
    #lat = 2 * np.arctan(np.exp(y/R)) - math.pi/2
    print(np.array((lng, lat)).T)
    return np.deg2rad(np.array((lng, lat)).T)


def to_cartesian(lnglat):
    '''
    r = 1
    '''
    lng = lnglat[:,0]
    lat = lnglat[:,1]
    x = np.sin(lat)*np.cos(lng)
    y = np.sin(lng)*np.sin(lat)
    z = np.cos(lat)
    return np.array((x, y, z)).T



#result = to_lnglat(np.array([(0, 0), (512, -256)]), 1024 / (2*math.pi))
def pixel2world(cam_pos, w, h):
    '''
    cam_pos = np.array([(x, y, z)])
    w = 1024
    h = 512
    '''
    #cam_pos = cam_pos.numpy()
    points = np.array([(i, j) for j in range(h) for i in range(w)])
    lnglat = to_lnglat(points, w, h)
    result = to_cartesian(lnglat)
    #result = result + cam_pos

    # Visualize
    #pcd = o3d.geometry.PointCloud()
    #pcd.points = o3d.utility.Vector3dVector(result)
    #o3d.visualization.draw_geometries([pcd])
    print("[p2w] ", result.shape)
    #result = np.transpose(result, axes=(1,0,2))
    result = np.expand_dims(result, axis=0) # (m, h, w, 3)
    result = torch.from_numpy(result)
    return result.to(torch.float32)
    #print(np.rad2deg(result))
    #print(result)

# pixel2world(np.array([(0, 0, 0)]), 1024, 512)


