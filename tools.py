import numpy as np

def circle_points(n):
    # generate n points within the unit disc

    theta = 2 * np.pi * np.random.rand(n)
    rho = np.sqrt(np.random.rand(n))

    polar = np.vstack((theta, rho)).T
    x = np.vstack((rho * np.cos(theta), rho * np.sin(theta))).T
    return polar, x

def anuular(n):
    # generate n points within the annular of radius 0.5~1 centered at [0,0]

    theta = 2 * np.pi * np.random.rand(n)
    rho = np.sqrt(np.random.uniform(0.5 ** 2, 1 ** 2, n))
    #rho = np.sqrt(np.random.rand(n)) / 2 + 0.5
    polar = np.vstack((theta, rho)).T
    x = np.vstack((rho * np.cos(theta), rho * np.sin(theta))).T
    return polar, x

def square(n):
    a = int(0.9*n)
    b = n - a
    x0 = np.random.rand(a,1)
    y0 = np.random.rand(a,1)
    x1 = 0.35 + 0.3 * np.random.rand(b,1)
    y1 = 1 + 0.3 * np.random.rand(b,1)
    pt0 = np.concatenate((x0, y0), axis=-1)
    pt1 = np.concatenate((x1, y1), axis=-1)
    pt = np.vstack((pt0, pt1))
    return pt

def gen_inner_pts(array, circle):
    center = get_barycenter(array)
    poly_0 = array - center
    circle_pts = circle
    circle_r = circle_pts[:,1]
    circle_theta = circle_pts[:,0]
    pts_r, pts_x = get_boundary_pts_theta(poly_0, circle_theta)
    if array.shape[1] == 3:
        poly_pts = np.concatenate((circle_r.reshape(-1,1) * pts_x, np.zeros((circle_r.shape[0],1))),axis=1) + center
    else:
        poly_pts = circle_r.reshape(-1, 1) * pts_x + center
    return poly_pts

def gen_inner_pts_annular(outer, inner, annular_pts):
    center = get_barycenter(inner)
    circle_pts = annular_pts
    circle_r = circle_pts[:,1]
    circle_theta = circle_pts[:,0]
    pts_r_outer, pts_x_outer = get_boundary_pts_annular_theta(inner, outer, circle_theta)
    pts_r_inner, pts_x_inner = get_boundary_pts_theta(inner, circle_theta)
    #t = (pts_r_outer - pts_r_inner) * (circle_r.reshape(-1,1) - r)/r
    pts_r = pts_r_inner + (pts_r_outer - pts_r_inner) * (circle_r.reshape(-1,1) - 0.5) / 0.5
    pts_x = pts_r * np.cos(circle_theta).reshape(-1,1)
    pts_y = pts_r * np.sin(circle_theta).reshape(-1, 1)
    pts = np.concatenate((pts_x, pts_y), axis=-1)
    if outer.shape[1] == 3:
        poly_pts = np.concatenate((pts, np.zeros((circle_r.shape[0],1))),axis=1) + center
    else:
        poly_pts = pts + center
    return poly_pts

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

def get_boundary_pts_annular_theta(inner, outer, theta):
    barycenter = get_barycenter(inner)
    outer = outer - barycenter
    inner_shift = np.roll(outer, shift=-1, axis=0)
    theta = theta.reshape(-1, 1)
    # r = ((x1 * (y2 - y1) - y1 * (x2 - x1)) / (cos(theta) * (y2 - y1) - sin(theta) * (x2 - x1)))
    x1, y1, x2, y2 = outer[:, 0], outer[:, 1], inner_shift[:, 0], inner_shift[:, 1]
    v1 = x1 * (y2 - y1) - y1 * (x2 - x1)
    v2 = np.cos(theta) @ (y2 - y1).reshape(1, -1) - np.sin(theta) @ (x2 - x1).reshape(1, -1)
    # exclude zero divisor
    parallel = np.abs(v2) < 1e-13
    sign = np.sign(v2[parallel])
    sign[sign == 0] = 1
    v2[parallel] = sign * 1e-13
    # compute intersection matrix
    R = v1 / v2  # [n, poly.shape[0]]
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

def gen_boundary_pts_annular(inner, outer, n_inner, n_outer):
    theta_outer = (2 * np.pi / n_outer) * np.arange(n_outer)[:, None]
    theta_inner = (2 * np.pi / n_inner) * np.arange(n_inner)[:, None]
    out_r, out_xy = get_boundary_pts_annular_theta(inner, outer, theta_outer)
    inner_r, inner_xy = get_boundary_pts_theta(inner, theta_inner)
    return np.concatenate((out_r, inner_r),axis=0), np.concatenate((out_xy, inner_xy),axis=0)

def interpolate_on_polygon_edges(poly, pts, g, num):
    center = get_barycenter(poly)
    pts = pts - center
    pts_r = np.linalg.norm(pts, axis = 1, keepdims=True)
    pts_sin = pts[:, 1:2]/pts_r
    sign = np.sign(pts[:,0:1])
    bias = np.pi * (1-sign) /2
    theta = (sign * np.arcsin(pts_sin) + bias) % (2*np.pi)
    theta[pts_sin == -1] = 3/2 *np.pi
    sample_theta = (2 * np.pi / num) * np.arange(num)
    mask = theta - sample_theta.reshape(1, -1)
    index = np.full((num), np.nan)
    for col in range(mask.shape[1]):
        column = mask[:, col]
        n = len(column)

        for i in range(n):
            next_i = (i + 1) % n
            if column[i] < 0 and column[next_i] > 0:
                index[col] = int(i)
    special = np.where(np.isnan(index))
    index = np.where(np.isnan(index), int(np.argmin(theta)-1), index)
    index = index.astype(int)
    theta_front = theta.reshape(-1)[index]
    theta_front[special] -= 2*np.pi
    theta_back = theta.reshape(-1)[(index + 1)%pts.shape[0]]
    g_front = g[index]
    g_back = g[(index + 1)%pts.shape[0]]
    g_sample = ((sample_theta - theta_front) * g_front + (theta_back - sample_theta) * g_back)/(theta_back - theta_front)
    return g_sample

def poly_2_circle(pts, poly):
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
    return circle_pts

def annular_2_circle(pts, outer, inner):
    center = get_barycenter(inner)
    pts = pts - center
    pts_r = np.linalg.norm(pts, axis=1, keepdims=True)
    pts_sin = pts[:, 1:2] / pts_r
    sign = np.sign(pts[:, 0:1])
    bias = np.pi * (1 - sign) / 2
    theta = (sign * np.arcsin(pts_sin) + bias) % (2 * np.pi)
    theta[pts_sin == -1] = 3 / 2 * np.pi
    r_out = get_boundary_pts_annular_theta(inner, outer, theta)[0]
    r_in = get_boundary_pts_theta(inner , theta)[0]
    r_p = 0.5 + 0.5 * ((pts_r - r_in)/(r_out - r_in))
    circle_pts = np.hstack((r_p * np.cos(theta), r_p * np.sin(theta)))
    return circle_pts

def square_2_area(pts, area, num):
    up = num - int(0.9 * num)
    pt_up = pts[-up:]
    l = 0.3
    pt_dire = pt_up - np.array([[0.35],[1]]).reshape(1,2)
    direction = pt_dire * l/0.3
    pt_area = area[2] + direction
    return np.vstack((pts[:-up], pt_area))

def area_2_square(pts, area):
    pt_rd = area[2][:2]
    l = area[3][1] - area[2][1]
    pt_uniform = []
    for i in range(pts.shape[0]):
        if pts[i][1] > 1:
            pi = np.array([[0.25],[1]]).reshape(1,2) + (pts[i][:2] - pt_rd) * 0.5/l
            pt_uniform.append(pi.reshape(-1))
        else:
            pt_uniform.append(pts[i][:2])
    return np.array(pt_uniform, dtype='float64').reshape(-1,2)

def triangle_inner_mumlti(pts, tri_index, pt):
    '''
    Input:
        pts: N * 3 array
        index: M * 3 array
        pt: t*3 array
    Output: 
        array t*1 of index of tri
    '''
    A = pts[tri_index[:, 0]]
    B = pts[tri_index[:, 1]]
    C = pts[tri_index[:, 2]]
    X_A, Y_A = A[:, 0].reshape(1, -1, 1), A[:, 1].reshape(1, -1, 1)
    X_B, Y_B = B[:, 0].reshape(1, -1, 1), B[:, 1].reshape(1, -1, 1)
    X_C, Y_C = C[:, 0].reshape(1, -1, 1), C[:, 1].reshape(1, -1, 1)
    deter_0 = ((Y_B - Y_A)*(pt[:,0].reshape(-1, 1, 1) - X_A) - (X_B - X_A)*(pt[:,1].reshape(-1, 1, 1) - Y_A))*((Y_B - Y_A)*(X_C - X_A) - (X_B - X_A)*(Y_C - Y_A))
    deter_1 = ((Y_C - Y_B)*(pt[:,0].reshape(-1, 1, 1) - X_B) - (X_C - X_B)*(pt[:,1].reshape(-1, 1, 1) - Y_B))*((Y_C - Y_B)*(X_A - X_B) - (X_C - X_B)*(Y_A - Y_B))
    deter_2 = ((Y_A - Y_C)*(pt[:,0].reshape(-1, 1, 1) - X_C) - (X_A - X_C)*(pt[:,1].reshape(-1, 1, 1) - Y_C))*((Y_A - Y_C)*(X_B - X_C) - (X_A - X_C)*(Y_B - Y_C))
    deter = np.concatenate((deter_0.reshape(pt.shape[0],-1,1), deter_1.reshape(pt.shape[0],-1,1), deter_2.reshape(pt.shape[0],-1,1)), axis = 2)
    check = np.all(deter >= 0, axis=2)
    index = np.argmax(check, axis=1)
    #index = np.where(check)[-1]
    index_check = np.where(check)[0].tolist()
    index_wrong = []
    for i in range(5000):
        if i not in index_check:
            index_wrong.append(i)
    return index

def solve_inter(solve, solve_pts, solve_tri_index, pts):
    index_inter = triangle_inner_mumlti(solve_pts, solve_tri_index, pts)
    if len(index_inter) == 0:
        return False
    elif len(index_inter) != pts.shape[0]:
        return False
    else:
        tri_inter = solve_tri_index[index_inter]
        index_A = tri_inter[:, 0]
        index_B = tri_inter[:, 1]
        index_C = tri_inter[:, 2]
        X_a, Y_a = solve_pts[index_A][:,0], solve_pts[index_A][:,1]
        X_b, Y_b = solve_pts[index_B][:,0], solve_pts[index_B][:,1]
        X_c, Y_c = solve_pts[index_C][:,0], solve_pts[index_C][:,1]
        #tri_pt = solve_pts[tri_inter[0]].reshape(-1,3)
        #triangle = mtri.Triangulation(solve_pts[:, 0], solve_pts[:, 1], solve_tri_index)
        X, Y = pts[:,0], pts[:,1]
        #b1 = ((X - X_a)*(Y_c - Y_a) - (X_c - X_a)*(Y - Y_a))/((X_b - X_a)*(Y_c - Y_a) - (X_c - X_a)*(Y_b - Y_a))
        #c1 = (Y - Y_a - (Y_b - Y_a)*b1)/(Y_c - Y_a)
        #a11 = 1-b1-c1
        a0 = np.concatenate(((X_a-X_c).reshape(-1,1,1), (X_b-X_c).reshape(-1,1,1)),axis=-1)
        a1 = np.concatenate(((Y_a-Y_c).reshape(-1,1,1), (Y_b-Y_c).reshape(-1,1,1)),axis=-1)
        s_a = np.concatenate((a0, a1), axis=1)
        para = np.linalg.solve(s_a, np.concatenate(((X-X_c).reshape(-1,1),(Y-Y_c).reshape(-1,1)), axis=-1)[..., None]).squeeze(-1)
        a = para[:,0]
        b = para[:,1]
        c = 1 - b - a
        u = a * solve[index_A] + b*solve[index_B] + c * solve[index_C]
        return u