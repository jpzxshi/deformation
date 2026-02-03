"""
@author: jpzxshi
"""
import numpy as np
from sklearn import gaussian_process as gp
from itertools import product
import pygmsh
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def generate_mesh(polygons, mesh_size=0.01):
    polygons = polygons.reshape(-1, polygons.shape[-2], polygons.shape[-1])
    mesh_data = {}
    for i in range(polygons.shape[0]):
        with pygmsh.geo.Geometry() as geom:
            geom.add_polygon(polygons[i], mesh_size=mesh_size)
            mesh = geom.generate_mesh()
        mesh_data['points_{}'.format(i)] = mesh.points
        mesh_data['vertex_{}'.format(i)] = mesh.cells[2].data
        mesh_data['line_{}'.format(i)] = mesh.cells[0].data
        mesh_data['triangle_{}'.format(i)] = mesh.cells[1].data
    return mesh_data

class Gaussian_process:
    '''Generate Gaussian process.
    '''
    def __init__(self, intervals, mean, std, length_scale, features, e=1e-13):
        self.intervals = intervals # e.g. [0, 1]
        self.mean = mean # e.g. 0
        self.std = std # e.g. 1
        self.length_scale = length_scale # e.g. 0.3
        self.features = features # e.g. 1000
        self.e = 1e-12

    def generate(self, num):
        if isinstance(self.intervals[0], list):
            itvs = []
            for interval in self.intervals:
                itvs.append(np.linspace(interval[0], interval[1], num=self.features))
            x = np.array(list(product(*itvs)))
            d = len(self.intervals)
        else:
            x = np.linspace(self.intervals[0], self.intervals[1], num=self.features)[:, None]
            d = 1
        A = gp.kernels.RBF(length_scale=self.length_scale)(x)
        L = np.linalg.cholesky(A + self.e * np.eye(x.shape[0]))
        res = (L @ np.random.randn(x.shape[0], num)).transpose() * self.std + self.mean # [num, features ** d]
        return res.reshape([num] + [self.features] * d)
    
def generate_mesh_high(mesh_size=0.01):
    n=1999
    
    mesh_low = np.load('mesh_6.npz')
    poly = mesh_low['points_{}'.format(n)][mesh_low['vertex_{}'.format(n)]].reshape(-1,3)
    print(poly.shape)
    
    mesh_high = generate_mesh(poly, mesh_size=mesh_size)
    points = mesh_high['points_0']
    vertex = mesh_high['vertex_0']
    line = mesh_high['line_0']
    triangle = mesh_high['triangle_0']
    
    print(points.shape)
    print(vertex.shape)
    print(line.shape)
    print(triangle.shape)
    
    vertex_points = points[vertex[:, 0]]
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangle)
    fig, ax = plt.subplots()
    ax.triplot(triangulation, color='black', linewidth=0.5)
    ax.set_aspect('equal')
    plt.scatter(vertex_points[:, 0], vertex_points[:, 1], s=3)
    plt.title('mesh')
    #plt.savefig('mesh.pdf')
    plt.show()
    return mesh_high
    
def generate_mesh_f(mesh):
    points = mesh['points_0']
    #vertex = mesh['vertex_0']
    #line = mesh['line_0']
    triangle = mesh['triangle_0']
    
    intervals = [[0,1]]*2 #[0,2*np.pi]
    mean = 0 # 1,0
    std = 1 #0.2, 1
    length_scale = 0.2 #0.2
    features = 100 #100
    
    gp = Gaussian_process(intervals, mean, std, length_scale, features)
    gps = gp.generate(1)
    
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    z = gps[0]
    func = interpolate.interp2d(x, y, z, kind='cubic')
    value = []
    for i in range(points.shape[0]):
        value.append(func(points[i][0], points[i][1]))
    f = np.array(value)
    print(f.shape)
    
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangle)
    fig, ax = plt.subplots()
    tpc = ax.tripcolor(triangulation, f.ravel(), shading ='gouraud', cmap='rainbow')
    ax.set_aspect('equal')
    fig.colorbar(tpc)
    plt.title('f')
    plt.show()
    
    np.savez_compressed('./mesh_{}'.format(points.shape[0]), **mesh) # mesh_482187
    np.save('./f_{}'.format(points.shape[0]), f) # mesh_482187
    
    return mesh, f

def test_mesh_f():
    mesh = np.load('./him_data/mesh_482187.npz') # './him_data/mesh_482187.npz'
    points = mesh['points_0']
    vertex = mesh['vertex_0']
    line = mesh['line_0']
    triangle = mesh['triangle_0']
    
    print(points.shape)
    print(vertex.shape)
    print(line.shape)
    print(triangle.shape)
    
    vertex_points = points[vertex[:, 0]]
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangle)
    fig, ax = plt.subplots()
    ax.triplot(triangulation, color='black', linewidth=0.5)
    ax.set_aspect('equal')
    plt.scatter(vertex_points[:, 0], vertex_points[:, 1], s=3)
    plt.title('mesh')
    plt.show()
    
    f = np.load('./him_data/f_482187.npy') # './him_data/f_482187.npy'
    print(f.shape)
    
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangle)
    fig, ax = plt.subplots()
    tpc = ax.tripcolor(triangulation, f.ravel(), shading ='gouraud', cmap='rainbow')
    ax.set_aspect('equal')
    fig.colorbar(tpc)
    plt.title('f')
    plt.show()

def main():
    #mesh_size = 0.001
    #mesh = generate_mesh_high(mesh_size)
    #generate_mesh_f(mesh)
    
    test_mesh_f()

if __name__ == '__main__':
    main()