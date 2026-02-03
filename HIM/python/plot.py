"""
@author: Pengzhan Jin (jpz@pku.edu.cn)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def plot_mesh():
    mesh = np.load('mesh_6.npz')  # 2000 examples
    #print(mesh.files)
    
    n = 0 # 0,1,2,...,1999
    points = mesh['points_{}'.format(n)]
    vertex = mesh['vertex_{}'.format(n)] # vertexes of polygon
    line = mesh['line_{}'.format(n)] # boundary lines of mesh
    triangle = mesh['triangle_{}'.format(n)]
    vertex_points = points[vertex[:, 0]]

    print(points.shape)
    print(points.dtype)
    #print(points)
    print(vertex.shape)
    #print(vertex)
    print(line.shape)
    #print(line)
    print(triangle.shape)
    #print(triangle)
    
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangle)
    fig, ax = plt.subplots()
    ax.triplot(triangulation, color='black', linewidth=0.5)
    plt.scatter(vertex_points[:, 0], vertex_points[:, 1], s=3)
    plt.title('mesh')
    #plt.savefig('mesh.pdf')
    plt.show()
    
def plot_f():
    mesh = np.load('mesh_6.npz')
    func_f = np.load('data_func_mesh_6.npz')
    
    n = 0 # 0,1,2,...,1999
    points = mesh['points_{}'.format(n)]
    triangle = mesh['triangle_{}'.format(n)]
    f = func_f['func_{}'.format(n)]
    
    print(points.shape)
    print(triangle.shape)
    print(f.shape)
    print(f.dtype)
    
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangle)
    fig, ax = plt.subplots()
    tpc = ax.tripcolor(triangulation, f.ravel(), shading ='gouraud', cmap='rainbow')
    fig.colorbar(tpc)
    plt.title('f')
    #plt.savefig('f.pdf')
    plt.show()
    
def plot_u():
    mesh = np.load('mesh_6.npz')
    solve_u = np.load('data_solve_mesh_6.npz')
    
    n = 0 # 0,1,2,...,1999
    points = mesh['points_{}'.format(n)]
    triangle = mesh['triangle_{}'.format(n)]
    u = solve_u['solve_{}'.format(n)]
    
    print(points.shape)
    print(triangle.shape)
    print(u.shape)
    
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangle)
    fig, ax = plt.subplots()
    tpc = ax.tripcolor(triangulation, u.ravel(), shading ='gouraud', cmap='rainbow')
    fig.colorbar(tpc)
    plt.title('u')
    #plt.savefig('u.pdf')
    plt.show()

def main():
    plot_mesh()
    plot_f()
    plot_u()


if __name__ == '__main__':
    main()