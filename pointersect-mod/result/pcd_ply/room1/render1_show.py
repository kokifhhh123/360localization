import numpy as np
import open3d as o3d

pcd = o3d.io.read_point_cloud("/home/koki/HoHoNet/render1.pcd")
print(pcd)
print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd])