import open3d as o3d
import numpy as np

def gen_sphere_ray(x, y, z):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    sphere.compute_vertex_normals()
    #o3d.visualization.draw_geometries([shpere])
    #600000
    pcd = sphere.sample_points_uniformly(number_of_points=600000)
    pcd.translate((x, y, z))
    #print(f'Center of pcd: {pcd.get_center()}')
    #o3d.visualization.draw_geometries([pcd])
    sphere_points = np.asarray(pcd.points)
    #print(np.asarray(pcd.points))
    #print(np.asarray(sphere.vertices))
    out = np.zeros((600000, 6))
    out[:, :3] = np.array([x, y, z])
    out[:, 3:] = np.subtract(sphere_points, np.array([x, y, z]))
    #print(out)
    out_o3d = o3d.core.Tensor(out.tolist(), dtype=o3d.core.Dtype.Float32)
    #print(out_o3d)
    return out_o3d

if __name__ == "__main__":
    '''
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    sphere.compute_vertex_normals()
    #o3d.visualization.draw_geometries([shpere])
    #600000
    pcd = sphere.sample_points_uniformly(number_of_points=600000)
    pcd.translate((2, 1, 2))
    #print(f'Center of pcd: {pcd.get_center()}')
    #o3d.visualization.draw_geometries([pcd])
    print(np.asarray(pcd.points).shape)
    #print(np.asarray(sphere.vertices))
    '''
    gen_sphere_ray(2, 1, 2)