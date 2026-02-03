"""
@author: jpzxshi
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

#### boundary
def get_barycenter(poly):
    '''
    poly: [N, 2] or [N, 3]
    '''
    poly_shift = np.roll(poly, shift=-1, axis=0)
    Barycenters = (poly + poly_shift) / 3
    Areas = (poly[:, 0] * poly_shift[:, 1] - poly[:, 1] * poly_shift[:, 0]) / 2
    return np.sum(Barycenters * Areas[:, None], axis=0) / np.sum(Areas)

def get_boundary_pts_theta(poly, theta):
    barycenter = get_barycenter(poly)
    poly = poly - barycenter
    poly_shift = np.roll(poly, shift=-1, axis=0)
    theta = theta.reshape(-1, 1)
    # r = ((x1 * (y2 - y1) - y1 * (x2 - x1)) / (cos(theta) * (y2 - y1) - sin(theta) * (x2 - x1)))
    x1, y1, x2, y2 = poly[:, 0], poly[:, 1], poly_shift[:, 0], poly_shift[:, 1]
    v1 = x1 * (y2 - y1) - y1 * (x2 - x1)
    v2 = np.cos(theta) @ (y2 - y1).reshape(1, -1) - np.sin(theta) @ (x2 - x1).reshape(1, -1)
    # exclude zero divisor
    parallel = np.abs(v2) < 1e-13
    sign = np.sign(v2[parallel])
    sign[sign == 0] = 1
    v2[parallel] = sign * 1e-13
    # compute intersection matrix
    R = v1 / v2 # [n, poly.shape[0]]
    R[R <= 0] = 1e13
    # find intersection points
    X, Y = R * np.cos(theta), R * np.sin(theta)
    check = ((x1 - X) * (x2 - X) + (y1 - Y) * (y2 - Y)) <= 0
    index = check.argmax(axis=1)
    inter_r = R[np.arange(theta.shape[0]), index][:, None]
    inter_x = np.hstack((inter_r * np.cos(theta), inter_r * np.sin(theta))) + barycenter[:2]
    return inter_r, inter_x

def get_boundary_pts(poly, n):
    theta = (2 * np.pi / n) * np.arange(n)[:, None]
    return get_boundary_pts_theta(poly, theta)
    
def boundary_test(n):
    mesh = np.load('mesh_6.npz')
    poly_index = mesh['vertex_{}'.format(n)]
    poly = mesh['points_{}'.format(n)][poly_index].reshape(-1,3)
    r, x = get_boundary_pts(poly, 200)
    print(r.shape, x.shape)
    
    #fig, ax = plt.subplots(figsize=[10, 10])
    plt.figure(figsize=[10, 10])
    plt.scatter(x[:, 0], x[:, 1], s=3)
    plt.xlim([0,1])
    plt.ylim([0,1])
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()

#### f
def get_inner_pts(poly, circle_pts_polar):
    center = get_barycenter(poly)
    poly_0 = poly - center
    circle_r = circle_pts_polar[:,1]
    circle_theta = circle_pts_polar[:,0]
    pts_r, pts_x = get_boundary_pts_theta(poly_0, circle_theta)
    if poly.shape[1] == 3:
        poly_pts = np.concatenate((circle_r.reshape(-1,1) * pts_x, np.zeros((circle_r.shape[0],1))),axis=1) + center
    else:
        poly_pts = circle_r.reshape(-1, 1) * pts_x + center
    return poly_pts

def get_triangle_index(points, tri_index, pts):
    '''
    INPUT
    points: N * 3 array
    tri_index: M * 3 array
    pts: t*3 array
    RETURN
    index: t*1 array of the indexes of triangles containing pts
    '''
    A = points[tri_index[:, 0]]
    B = points[tri_index[:, 1]]
    C = points[tri_index[:, 2]]
    X_A, Y_A = A[:, 0].reshape(1, -1, 1), A[:, 1].reshape(1, -1, 1)
    X_B, Y_B = B[:, 0].reshape(1, -1, 1), B[:, 1].reshape(1, -1, 1)
    X_C, Y_C = C[:, 0].reshape(1, -1, 1), C[:, 1].reshape(1, -1, 1)
    deter_0 = ((Y_B - Y_A)*(pts[:,0].reshape(-1, 1, 1) - X_A) - (X_B - X_A)*(pts[:,1].reshape(-1, 1, 1) - Y_A))*((Y_B - Y_A)*(X_C - X_A) - (X_B - X_A)*(Y_C - Y_A))
    deter_1 = ((Y_C - Y_B)*(pts[:,0].reshape(-1, 1, 1) - X_B) - (X_C - X_B)*(pts[:,1].reshape(-1, 1, 1) - Y_B))*((Y_C - Y_B)*(X_A - X_B) - (X_C - X_B)*(Y_A - Y_B))
    deter_2 = ((Y_A - Y_C)*(pts[:,0].reshape(-1, 1, 1) - X_C) - (X_A - X_C)*(pts[:,1].reshape(-1, 1, 1) - Y_C))*((Y_A - Y_C)*(X_B - X_C) - (X_A - X_C)*(Y_B - Y_C))
    deter = np.concatenate((deter_0.reshape(pts.shape[0],-1,1), deter_1.reshape(pts.shape[0],-1,1), deter_2.reshape(pts.shape[0],-1,1)), axis = 2)
    check = np.all(deter >= 0, axis=2)
    index = np.where(check)[-1]
    return index

def interpolate(points, tri_index, f, pts):
    index_inter = get_triangle_index(points, tri_index, pts)
    if len(index_inter) == 0:
        raise ValueError
    else:
        tri_inter = tri_index[index_inter]
        index_A = tri_inter[:, 0]
        index_B = tri_inter[:, 1]
        index_C = tri_inter[:, 2]
        X_a, Y_a = points[index_A][:,0], points[index_A][:,1]
        X_b, Y_b = points[index_B][:,0], points[index_B][:,1]
        X_c, Y_c = points[index_C][:,0], points[index_C][:,1]
        X, Y = pts[:,0], pts[:,1]
        b = ((X - X_a)*(Y_c - Y_a) - (X_c - X_a)*(Y - Y_a))/((X_b - X_a)*(Y_c - Y_a) - (X_c - X_a)*(Y_b - Y_a))
        c = (Y - Y_a - (Y_b - Y_a)*b)/(Y_c - Y_a)
        a = 1-b-c
        f = f.ravel()
        values = a * f[index_A] + b * f[index_B] + c * f[index_C]
        return values
    
def f_test(n):
    mesh = np.load('mesh_6.npz')
    data_f = np.load('data_func_mesh_6.npz')
    #data_u = np.load('data_solve_mesh_6.npz')
    data_sample_points = np.load('data_random_unit_circle.npy') # theta, r
    #net = torch.load('model_best.pkl')
    
    points = mesh['points_{}'.format(n)]
    vertex = mesh['vertex_{}'.format(n)]
    line = mesh['line_{}'.format(n)]
    triangle = mesh['triangle_{}'.format(n)]
    f = data_f['func_{}'.format(n)]
    
    print(points.shape)
    print(vertex.shape)
    print(line.shape)
    print(triangle.shape)
    print(f.shape)
    print(data_sample_points.shape)
    
    poly = points[vertex].reshape(-1,3)
    inter_pts = get_inner_pts(poly, data_sample_points)
    
    print(inter_pts.shape)
    
    #cx = data_sample_points[:, 1] * np.cos(data_sample_points[:, 0])
    #cy = data_sample_points[:, 1] * np.sin(data_sample_points[:, 0])
    plt.figure(figsize=[10, 10])
    plt.scatter(inter_pts[:, 0], inter_pts[:, 1], s=1)
    #plt.scatter(cx, cy, s=1)
    plt.xlim([0,1])
    plt.ylim([0,1])
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()
    
    inter_f = interpolate(points, triangle, f, inter_pts)
    print(inter_f.shape)
    
    plt.figure(figsize=[10, 10])
    plt.scatter(inter_pts[:, 0], inter_pts[:, 1], c=inter_f, s=1)
    plt.xlim([0,1])
    plt.ylim([0,1])
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()
    
#### model
def poly_2_circle(poly, pts):
    center = get_barycenter(poly)
    pts = pts - center
    pts_r = np.linalg.norm(pts, axis = 1, keepdims=True)
    pts_sin = pts[:, 1:2]/pts_r
    sign = np.sign(pts[:,0:1])
    bias = np.pi * (1-sign) /2
    theta = (sign * np.arcsin(pts_sin) + bias) % (2*np.pi)
    theta[pts_sin == -1] = 3/2 *np.pi
    r = get_boundary_pts_theta(poly - center, theta)[0]
    circle_pts = np.hstack((pts_r/r * np.cos(theta), pts_r/r * np.sin(theta)))
    #circle_pts = np.hstack((theta.reshape(-1,1), pts_r.reshape(-1,1)/r))
    return circle_pts

def model_test(n):
    mesh = np.load('mesh_6.npz')
    data_f = np.load('data_func_mesh_6.npz')
    data_u = np.load('data_solve_mesh_6.npz')
    data_sample_points = np.load('data_random_unit_circle.npy') # theta, r
    
    points = mesh['points_{}'.format(n)]
    vertex = mesh['vertex_{}'.format(n)]
    #line = mesh['line_{}'.format(n)]
    triangle = mesh['triangle_{}'.format(n)]
    f = data_f['func_{}'.format(n)]
    u = data_u['solve_{}'.format(n)]
    
    #poly_index = mesh['vertex_{}'.format(n)]
    poly = mesh['points_{}'.format(n)][vertex].reshape(-1,3)
    r, x = get_boundary_pts(poly, 200)
    r = r.ravel()
    
    inter_pts = get_inner_pts(poly, data_sample_points)
    inter_f = interpolate(points, triangle, f, inter_pts)
    
    #circle_pts = poly_2_circle(poly, points)
    X = (points - get_barycenter(poly))[:, :2]
    print(r.shape, inter_f.shape, X.shape)
    
    net = torch.load('model_best.pkl')
    
    u_pred = net.predict((r, inter_f, X), returnnp=True).ravel() #????
    
    #import time
    #Y = torch.ones([490000, 2], device='cuda', dtype=torch.double)
    #t = time.time()
    #print(t)
    #result = net.predict((r, inter_f, Y), returnnp=False)
    #print(result[:5])
    #print(time.time() - t)
    
    
    print(u.shape, u_pred.shape)
    
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangle)
    fig, ax = plt.subplots()
    tpc = ax.tripcolor(triangulation, u, shading ='gouraud', cmap='rainbow')
    fig.colorbar(tpc)
    plt.title('u')
    
    fig, ax = plt.subplots()
    tpc = ax.tripcolor(triangulation, u_pred, shading ='gouraud', cmap='rainbow')
    fig.colorbar(tpc)
    plt.title('u_pred')
    
    #plt.savefig('prediction.pdf')
    plt.show()
    
    
    
def main():
    n = 1999
    #boundary_test(n)
    #f_test(n)
    model_test(n)

if __name__ == '__main__':
    main()