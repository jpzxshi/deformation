import os
import numpy as np
import pygmsh
from scipy.interpolate import RegularGridInterpolator#, interp1d
from gaussian_process import Gaussian_process#, Gaussian_process_period
from poisson_equation_solver import solve_poisson_equation
from plot import plot_mesh, plot_f, plot_u

def random_circle(center=[0, 0], radius=1, n=1):
    theta = 2 * np.pi * np.random.rand(n)
    rho = np.sqrt(np.random.rand(n))
    x = np.vstack((rho * np.cos(theta), rho * np.sin(theta))).T

    x = x * radius + np.array(center)
    return x

def generate_polygon(vertex_num, n, perturbation_rate=0.2):
    if vertex_num == 4:
        theta_init = 1 / 4 * np.pi
    elif vertex_num == 5:
        theta_init = 1 / 10 * np.pi
    else:
        theta_init = 0
    theta = theta_init + np.arange(vertex_num) * (2 * np.pi / vertex_num)
    baseline = np.vstack((np.cos(theta), np.sin(theta))).T
    points = np.hstack([random_circle(point, perturbation_rate, n) for point in baseline])
    shape = [n, vertex_num, 2] if n > 1 else [vertex_num, 2]
    return points.reshape(shape) * (0.5 / (1 + perturbation_rate)) + np.array([0.5, 0.5])

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
    y = np.linspace(0, 1, func.shape[1])
    Z = func
    f = RegularGridInterpolator((x, y), Z, method='linear')
    return f(pts)

def generate_data(n):
    # generate 3*n datapoints

    # generate area
    areas_4 = generate_polygon(4, n)# [n, 4, 2]
    areas_5 = generate_polygon(5, n)# [n, 5, 2]
    areas_6 = generate_polygon(6, n)# [n, 6, 2]

    # generate mesh
    meshes_4 = generate_mesh(areas_4, 0.01)
    meshes_5 = generate_mesh(areas_5, 0.01)
    meshes_6 = generate_mesh(areas_6, 0.01)

    # generate random function
    gpf = Gaussian_process([[0, 1]] * 2, 0, 1, 0.2, 100)
    f = gpf.generate(3*n)  # [n, 100, 100]

    # interpolation
    f_mesh_4 = []
    f_mesh_5 = []
    f_mesh_6 = []
    for i in range(n):
        pts_4 = meshes_4['points_{}'.format(i)]
        f_mesh_4.append(func_inter_rec(pts_4, f[i]))
        pts_5 = meshes_5['points_{}'.format(i)]
        f_mesh_5.append(func_inter_rec(pts_5, f[n+i]))
        pts_6 = meshes_6['points_{}'.format(i)]
        f_mesh_6.append(func_inter_rec(pts_6, f[2*n+i]))

    # solve
    solutions_4 = []
    solutions_5 = []
    solutions_6 = []
    for i in range(n):
        print('Solving PDE No. {} ...'.format(i))
        mesh4 = {'elements': meshes_4['triangle_{}'.format(i)],
                'points': meshes_4['points_{}'.format(i)],
                'line': meshes_4['line_{}'.format(i)]}
        solutions_4.append(
            solve_poisson_equation(mesh4, np.ones_like(f_mesh_4[i]), f_mesh_4[i], np.zeros(mesh4['line'].shape[0])))

        mesh5 = {'elements': meshes_5['triangle_{}'.format(i)],
                 'points': meshes_5['points_{}'.format(i)],
                 'line': meshes_5['line_{}'.format(i)]}
        solutions_5.append(
            solve_poisson_equation(mesh5, np.ones_like(f_mesh_5[i]), f_mesh_5[i], np.zeros(mesh5['line'].shape[0])))

        mesh6 = {'elements': meshes_6['triangle_{}'.format(i)],
                 'points': meshes_6['points_{}'.format(i)],
                 'line': meshes_6['line_{}'.format(i)]}
        solutions_6.append(
            solve_poisson_equation(mesh6, np.ones_like(f_mesh_6[i]), f_mesh_6[i], np.zeros(mesh6['line'].shape[0])))

    # data
    data_4 = {}
    data_5 = {}
    data_6 = {}
    data_4['num'] = n
    data_5['num'] = n
    data_6['num'] = n
    for i in range(n):
        data_4['points_{}'.format(i)] = meshes_4['points_{}'.format(i)]
        data_4['vertex_{}'.format(i)] = meshes_4['vertex_{}'.format(i)]
        data_4['line_{}'.format(i)] = meshes_4['line_{}'.format(i)]
        data_4['triangle_{}'.format(i)] = meshes_4['triangle_{}'.format(i)]
        data_4['f_{}'.format(i)] = f_mesh_4[i]
        data_4['u_{}'.format(i)] = solutions_4[i]

        data_5['points_{}'.format(i)] = meshes_5['points_{}'.format(i)]
        data_5['vertex_{}'.format(i)] = meshes_5['vertex_{}'.format(i)]
        data_5['line_{}'.format(i)] = meshes_5['line_{}'.format(i)]
        data_5['triangle_{}'.format(i)] = meshes_5['triangle_{}'.format(i)]
        data_5['f_{}'.format(i)] = f_mesh_5[i]
        data_5['u_{}'.format(i)] = solutions_5[i]

        data_6['points_{}'.format(i)] = meshes_6['points_{}'.format(i)]
        data_6['vertex_{}'.format(i)] = meshes_6['vertex_{}'.format(i)]
        data_6['line_{}'.format(i)] = meshes_6['line_{}'.format(i)]
        data_6['triangle_{}'.format(i)] = meshes_6['triangle_{}'.format(i)]
        data_6['f_{}'.format(i)] = f_mesh_6[i]
        data_6['u_{}'.format(i)] = solutions_6[i]

    return data_4, data_5, data_6


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


def save_data(data, vertex_num):
    save_dir = './data/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.savez_compressed(save_dir + '/poly_{}_raw_data.npz'.format(vertex_num), **data)
    data = np.load(save_dir + '/poly_{}_raw_data.npz'.format(vertex_num))
    #print(data.files)
    for i in range(min(data['num'], 2)):
        print('points_{}'.format(i), data['points_{}'.format(i)].shape, data['points_{}'.format(i)].dtype)
        print('vertex_{}'.format(i), data['vertex_{}'.format(i)].shape, data['vertex_{}'.format(i)].dtype)
        print('line_{}'.format(i), data['line_{}'.format(i)].shape, data['line_{}'.format(i)].dtype)
        print('triangle_{}'.format(i), data['triangle_{}'.format(i)].shape, data['triangle_{}'.format(i)].dtype)
        print('f_{}'.format(i), data['f_{}'.format(i)].shape, data['f_{}'.format(i)].dtype)
        print('u_{}'.format(i), data['u_{}'.format(i)].shape, data['u_{}'.format(i)].dtype)

def main():
    data4, data5, data6 = generate_data(2000)
    plot(data4)
    plot(data5)
    plot(data6)
    save_data(data4, 4)
    save_data(data5, 5)
    save_data(data6, 6)

if __name__ == '__main__':
    main()