import numpy as np
from tools import circle_points, get_boundary_pts, gen_inner_pts, interpolate_on_polygon_edges
from tools import anuular, gen_inner_pts_annular, square, square_2_area, solve_inter
import os

def poly_domain_d2e_data(n_train, n_test):
    num_sensor = 5000
    boundary_sensor = 200
    num_train = n_train
    num_test = n_test
    num = num_train + num_test

    raw_data_4 = np.load('./data/poly_4_raw_data.npz')
    raw_data_5 = np.load('./data/poly_5_raw_data.npz')
    raw_data_6 = np.load('./data/poly_6_raw_data.npz')
    circle_sample, circle_sample_xy = circle_points(num_sensor)
    boundary_sample = []
    f = []
    y = []
    inner_pts = []
    for raw_data in [raw_data_4, raw_data_5, raw_data_6]:
        for i in range(num):
            print('Processing No. {} ...'.format(i))
            points = raw_data['points_{}'.format(i)]
            vertex = raw_data['vertex_{}'.format(i)]
            tri = raw_data['triangle_{}'.format(i)]
            poly = points[vertex].reshape(-1, 2)
            random_pts = gen_inner_pts(poly, circle_sample)
            # boundary sample
            bi, b_plot = get_boundary_pts(poly, boundary_sensor)
            boundary_sample.append(bi)
            # f sample
            fi = solve_inter(raw_data['f_{}'.format(i)], points, tri, random_pts)
            f.append(fi)
            # inner points, y sample
            n = points.shape[0]
            if n > num_sensor:
                indices = np.random.choice(n, num_sensor, replace=False)
            else:
                indices = np.concatenate([np.arange(n), np.random.choice(n, num_sensor - n, replace=True)])
            ptsi = points[indices]
            yi = raw_data['u_{}'.format(i)][indices]
            inner_pts.append(ptsi)
            y.append(yi)

    boundary_sample = np.array(boundary_sample).reshape(3 * num, boundary_sensor)
    f = np.array(f).reshape(3 * num, num_sensor)
    y = np.array(y).reshape(3 * num, num_sensor)
    inner_pts = np.array(inner_pts).reshape(3 * num, num_sensor, 2)
    
    index_train = np.hstack((np.arange(0 * num, 0 * num + num_train), 
                             np.arange(1 * num, 1 * num + num_train),
                             np.arange(2 * num, 2 * num + num_train)))
    index_test = np.hstack((np.arange(1 * num - num_test, 1 * num), 
                             np.arange(2 * num - num_test, 2 * num),
                             np.arange(3 * num - num_test, 3 * num)))

    x_train = (boundary_sample[index_train], f[index_train], inner_pts[index_train])
    x_test = (boundary_sample[index_test], f[index_test], inner_pts[index_test])
    y_train = y[index_train]
    y_test = y[index_test]

    save_dir = './data/poly_d2e/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.savez_compressed(save_dir + '/X_train', *x_train)
    np.savez_compressed(save_dir + '/X_test', *x_test)
    np.save(save_dir + '/y_train', y_train)
    np.save(save_dir + '/y_test', y_test)
    np.save(save_dir + '/sample_points', circle_sample_xy)
    np.save(save_dir + '/sample_points_polar', circle_sample)

def poly_domain_d2d_data(n_train, n_test):
    num_sensor = 5000
    boundary_sensor = 200
    num_train = n_train
    num_test = n_test
    num = num_train + num_test

    raw_data_4 = np.load('./data/poly_4_raw_data.npz')
    raw_data_5 = np.load('./data/poly_5_raw_data.npz')
    raw_data_6 = np.load('./data/poly_6_raw_data.npz')
    circle_sample, circle_sample_xy = circle_points(num_sensor)
    boundary_sample = []
    f = []
    y = []
    for raw_data in [raw_data_4, raw_data_5, raw_data_6]:
        for i in range(num):
            print('Processing No. {} ...'.format(i))
            points = raw_data['points_{}'.format(i)]
            vertex = raw_data['vertex_{}'.format(i)]
            tri = raw_data['triangle_{}'.format(i)]
            poly = points[vertex].reshape(-1, 2)
            random_pts = gen_inner_pts(poly, circle_sample)
            # boundary sample
            bi, b_plot = get_boundary_pts(poly, boundary_sensor)
            boundary_sample.append(bi)
            # f, y sample
            fi = solve_inter(raw_data['f_{}'.format(i)], points, tri, random_pts)
            yi = solve_inter(raw_data['u_{}'.format(i)], points, tri, random_pts)
            f.append(fi)
            y.append(yi)

    boundary_sample = np.array(boundary_sample).reshape(3 * num, boundary_sensor)
    f = np.array(f).reshape(3 * num, num_sensor)
    y = np.array(y).reshape(3 * num, num_sensor)
    
    index_train = np.hstack((np.arange(0*num, 0*num+num_train), 
                             np.arange(1*num, 1*num+num_train),
                             np.arange(2*num, 2*num+num_train)))
    index_test = np.hstack((np.arange(1*num-num_test, 1*num), 
                             np.arange(2*num-num_test, 2*num),
                             np.arange(3*num-num_test, 3*num)))

    x_train = (boundary_sample[index_train], f[index_train], circle_sample_xy)
    x_test = (boundary_sample[index_test], f[index_test], circle_sample_xy)
    y_train = y[index_train]
    y_test = y[index_test]

    save_dir = './data/poly_d2d/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.savez_compressed(save_dir + '/X_train', *x_train)
    np.savez_compressed(save_dir + '/X_test', *x_test)
    np.save(save_dir + '/y_train', y_train)
    np.save(save_dir + '/y_test', y_test)
    np.save(save_dir + '/sample_points', circle_sample_xy)
    np.save(save_dir + '/sample_points_polar', circle_sample)

def star_domain_d2d_data(n_train, n_test):
    num_sensor = 5000
    boundary_sensor = 200
    num_train = n_train
    num_test = n_test

    raw_data = np.load('./data/star_raw_data.npz')
    circle_sample, circle_sample_xy = circle_points(num_sensor)
    boundary_sample = []
    f = []
    g = []
    k = []
    y = []
    for i in range(num_train + num_test):
        print('Processing No. {} ...'.format(i))
        points = raw_data['points_{}'.format(i)]
        vertex = raw_data['vertex_{}'.format(i)]
        tri = raw_data['triangle_{}'.format(i)]
        poly = points[vertex].reshape(-1,2)
        line = raw_data['line_{}'.format(i)]
        pts_b = points[line[:, 0]]
        random_pts = gen_inner_pts(poly, circle_sample)
        #boundary sample
        bi, b_plot = get_boundary_pts(poly, boundary_sensor)
        boundary_sample.append(bi)
        #f, k, y sample
        fi = solve_inter(raw_data['f_{}'.format(i)], points, tri, random_pts)
        ki = solve_inter(raw_data['k_{}'.format(i)], points, tri, random_pts)
        yi = solve_inter(raw_data['u_{}'.format(i)], points, tri, random_pts)
        f.append(fi)
        k.append(ki)
        y.append(yi)
        # g sample
        gi = interpolate_on_polygon_edges(poly, pts_b, raw_data['g_{}'.format(i)], boundary_sensor)
        g.append(gi)
    boundary_sample = np.array(boundary_sample).reshape(num_train + num_test, boundary_sensor)
    f = np.array(f).reshape(num_train + num_test, num_sensor)
    k = np.array(k).reshape(num_train + num_test, num_sensor)
    y = np.array(y).reshape(num_train + num_test, num_sensor)
    g = np.array(g).reshape(num_train + num_test, boundary_sensor)

    fg = np.concatenate((f, g), axis=-1)
    x_train = (boundary_sample[:num_train], fg[:num_train], k[:num_train], circle_sample_xy)
    x_test = (boundary_sample[-num_test:], fg[-num_test:], k[-num_test:], circle_sample_xy)
    y_train = y[:num_train]
    y_test = y[-num_test:]

    save_dir = './data/star_d2d/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.savez_compressed(save_dir + '/X_train', *x_train)
    np.savez_compressed(save_dir + '/X_test', *x_test)
    np.save(save_dir + '/y_train', y_train)
    np.save(save_dir + '/y_test', y_test)
    np.save(save_dir + '/sample_points', circle_sample_xy)
    np.save(save_dir + '/sample_points_polar', circle_sample)
    
def square_domain_d2e_data(n_train, n_test):
    num_sensor = 5000
    num_train = n_train
    num_test = n_test

    raw_data = np.load('./data/square_raw_data.npz')
    square_sample = square(num_sensor)
    centers = []
    f = []
    y = []
    inner_pts = []
    for i in range(num_train + num_test):
        print('Processing No. {} ...'.format(i))
        points = raw_data['points_{}'.format(i)]
        vertex = raw_data['vertex_{}'.format(i)]
        tri = raw_data['triangle_{}'.format(i)]
        poly = points[vertex].reshape(-1,2)
        #line = raw_data['line_{}'.format(i)]
        random_pts = square_2_area(square_sample, poly, num_sensor)
        # centers
        centers.append(np.mean(poly[2:6], axis=0)[:1])
        # f sample
        fi = solve_inter(raw_data['f_{}'.format(i)], points, tri, random_pts)
        f.append(fi)
        # inner points, y sample
        n = points.shape[0]
        if n > num_sensor:
            indices = np.random.choice(n, num_sensor, replace=False)
        else:
            indices = np.concatenate([np.arange(n), np.random.choice(n, num_sensor - n, replace=True)])
        ptsi = points[indices]
        yi = raw_data['u_{}'.format(i)][indices]
        inner_pts.append(ptsi)
        y.append(yi)
    
    centers = np.array(centers).reshape(num_train + num_test, 1)
    f = np.array(f).reshape(num_train + num_test, num_sensor)
    y = np.array(y).reshape(num_train + num_test, num_sensor)
    inner_pts = np.array(inner_pts).reshape(num_train + num_test, num_sensor, 2)

    x_train = (centers[:num_train], f[:num_train], inner_pts[:num_train])
    x_test = (centers[-num_test:], f[-num_test:], inner_pts[-num_test:])
    y_train = y[:num_train]
    y_test = y[-num_test:]

    save_dir = './data/square_d2e/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.savez_compressed(save_dir + '/X_train', *x_train)
    np.savez_compressed(save_dir + '/X_test', *x_test)
    np.save(save_dir + '/y_train', y_train)
    np.save(save_dir + '/y_test', y_test)
    np.save(save_dir + '/sample_points', square_sample)

def annular_domain_d2d_data(n_train, n_test):
    num_sensor = 5000
    boundary_sensor_out = 200
    boundary_sensor_in = 50
    boundary_sensor = boundary_sensor_out + boundary_sensor_in
    num_train = n_train
    num_test = n_test

    raw_data = np.load('./data/annular_raw_data.npz')
    annular_sample, annular_sample_xy = anuular(num_sensor)
    outer = raw_data['out']
    inner = raw_data['in']
    boundary_sample = []
    f = []
    #k = []
    y = []
    for i in range(num_train + num_test):
        print('Processing No. {} ...'.format(i))
        points = raw_data['points_{}'.format(i)]
        #vertex = raw_data['vertex_{}'.format(i)]
        tri = raw_data['triangle_{}'.format(i)]
        #poly = points[vertex].reshape(-1,2)
        #line = raw_data['line_{}'.format(i)]
        #pts_b = points[line[:, 0]]
        random_pts = gen_inner_pts_annular(outer[i], inner[i], annular_sample)
        #boundary sample
        bi_out, b_plot = get_boundary_pts(outer[i], boundary_sensor_out)
        bi_in, _ = get_boundary_pts(inner[i], boundary_sensor_in)
        bi = np.concatenate((bi_out, bi_in), axis=0)
        boundary_sample.append(bi)
        #f, k, y sample
        fi = solve_inter(raw_data['f_{}'.format(i)], points, tri, random_pts)
        #ki = solve_inter(raw_data['k_{}'.format(i)], points, tri, random_pts)
        yi = solve_inter(raw_data['u_{}'.format(i)], points, tri, random_pts)
        f.append(fi)
        #k.append(ki)
        y.append(yi)
    boundary_sample = np.array(boundary_sample).reshape(num_train + num_test, boundary_sensor)
    f = np.array(f).reshape(num_train + num_test, num_sensor)
    #k = np.array(k).reshape(num_train + num_test, num_sensor)
    y = np.array(y).reshape(num_train + num_test, num_sensor)


    x_train = (boundary_sample[:num_train], f[:num_train], annular_sample_xy)
    x_test = (boundary_sample[-num_test:], f[-num_test:], annular_sample_xy)
    y_train = y[:num_train]
    y_test = y[-num_test:]

    save_dir = './data/annular_d2d/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.savez_compressed(save_dir + '/X_train', *x_train)
    np.savez_compressed(save_dir + '/X_test', *x_test)
    np.save(save_dir + '/y_train', y_train)
    np.save(save_dir + '/y_test', y_test)
    np.save(save_dir + '/sample_points', annular_sample_xy)
    np.save(save_dir + '/sample_points_polar', annular_sample)

def main():
    # If an out-of-memory error occurs, the amount of test data can be reduced, 
    # and training with small batch size.
    
    poly_domain_d2e_data(n_train=1500, n_test=500)
    #poly_domain_d2d_data(n_train=1500, n_test=500)
    #star_domain_d2d_data(n_train=2500, n_test=500)
    #square_domain_d2e_data(n_train=3500, n_test=500)
    #annular_domain_d2d_data(n_train=2500, n_test=500)

if __name__ == '__main__':
    main()