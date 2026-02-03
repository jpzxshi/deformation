import os
import numpy as np
import pygmsh
from scipy.interpolate import RegularGridInterpolator#, interp1d
from gaussian_process import Gaussian_process
from poisson_equation_solver import solve_poisson_equation
from plot import plot_mesh, plot_f, plot_u

# generate local changed square domains
def generate_square(num):
    x2_random = 0.15 + 0.4 * np.random.rand(num,1)
    x2 = np.concatenate((x2_random, np.ones_like(x2_random)), axis=-1).reshape(num, 1, 2)
    l_random = 0.3
    x3 = np.concatenate((x2_random, (1+l_random) * np.ones_like(x2_random)), axis=-1).reshape(num, 1, 2)
    x4 = np.concatenate((l_random + x2_random, (1+l_random) * np.ones_like(x2_random)), axis=-1).reshape(num, 1, 2)
    x5 = np.concatenate((l_random + x2_random, np.ones_like(x2_random)), axis=-1).reshape(num, 1, 2)
    x0 = np.zeros_like(x2)
    x1 = np.concatenate((np.zeros_like(x2_random),np.ones_like(x2_random)), axis=-1).reshape(num, 1, 2)
    x6 = np.ones_like(x2).reshape(num, 1, 2)
    x7 = np.concatenate((np.ones_like(x2_random),np.zeros_like(x2_random)), axis=-1).reshape(num, 1, 2)
    area = np.concatenate((x0, x1, x2, x3, x4, x5, x6, x7), axis=1)
    return area

# generate mesh given polygons
def generate_mesh(polygons, mesh_size = 0.01):
    polygons = polygons.reshape(-1, polygons.shape[-2], polygons.shape[-1])
    data = {}
    for i in range(polygons.shape[0]):
        print('Generating mesh No. {} ...'.format(i))
        with pygmsh.geo.Geometry() as geom:
            geom.add_polygon(polygons[i], mesh_size)
            mesh = geom.generate_mesh()
        data['points_{}'.format(i)] = mesh.points[:, :2]
        data['vertex_{}'.format(i)] = mesh.cells[2].data
        data['line_{}'.format(i)] = mesh.cells[0].data
        data['triangle_{}'.format(i)] = mesh.cells[1].data
    return data

# interpolate for function on [0,1]^2
def func_inter_rec(pts, func):
    x = np.linspace(0, 1, func.shape[0])
    y = np.linspace(0, 1.3, func.shape[1])
    Z = func
    f = RegularGridInterpolator((x, y), Z, method='linear')
    return f(pts)


def generate_data(n):
    # generate n datapoints

    # generate area
    areas = generate_square(n)  # [n, 8, 2]

    # generate mesh
    meshes = generate_mesh(areas, 0.02)

    # generate random function
    gpf = Gaussian_process([[0, 1]] * 2, 0, 1, 0.2, 100)
    f = gpf.generate(n)  # [n, 100, 100]

    # interpolation
    f_mesh = []
    for i in range(n):
        pts = meshes['points_{}'.format(i)]
        f_mesh.append(func_inter_rec(pts, f[i]))

    # solve
    solutions = []
    for i in range(n):
        print('Solving PDE No. {} ...'.format(i))
        mesh = {'elements': meshes['triangle_{}'.format(i)],
                    'points': meshes['points_{}'.format(i)],
                    'line': meshes['line_{}'.format(i)]}
        solutions.append(
            solve_poisson_equation(mesh, np.ones_like(f_mesh[i]), f_mesh[i], np.zeros(mesh['line'].shape[0])))

    # data
    data = {}
    data['num'] = n
    for i in range(n):
        data['points_{}'.format(i)] = meshes['points_{}'.format(i)]
        data['vertex_{}'.format(i)] = meshes['vertex_{}'.format(i)]
        data['line_{}'.format(i)] = meshes['line_{}'.format(i)]
        data['triangle_{}'.format(i)] = meshes['triangle_{}'.format(i)]
        data['f_{}'.format(i)] = f_mesh[i]
        data['u_{}'.format(i)] = solutions[i]

    return data

def plot(data):
    n = 0
    # plot
    mesh = {}
    mesh['points'] = data['points_{}'.format(n)]
    mesh['vertex'] = data['vertex_{}'.format(n)]
    mesh['line'] = data['line_{}'.format(n)]
    mesh['triangle'] = data['triangle_{}'.format(n)]
    f = data['f_{}'.format(n)]
    u = data['u_{}'.format(n)]

    plot_mesh(mesh)
    plot_f(mesh, f)
    plot_u(mesh, u)

def save_data(data):
    save_dir = './data/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.savez_compressed(save_dir + '/square_raw_data.npz', **data)
    data = np.load(save_dir + '/square_raw_data.npz')
    #print(data.files)
    for i in range(min(data['num'], 2)):
        print('points_{}'.format(i), data['points_{}'.format(i)].shape, data['points_{}'.format(i)].dtype)
        print('vertex_{}'.format(i), data['vertex_{}'.format(i)].shape, data['vertex_{}'.format(i)].dtype)
        print('line_{}'.format(i), data['line_{}'.format(i)].shape, data['line_{}'.format(i)].dtype)
        print('triangle_{}'.format(i), data['triangle_{}'.format(i)].shape, data['triangle_{}'.format(i)].dtype)
        print('f_{}'.format(i), data['f_{}'.format(i)].shape, data['f_{}'.format(i)].dtype)
        print('u_{}'.format(i), data['u_{}'.format(i)].shape, data['u_{}'.format(i)].dtype)

def main():
    data = generate_data(4000)
    plot(data)
    save_data(data)

if __name__ == '__main__':
    main()