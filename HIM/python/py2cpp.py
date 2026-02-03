"""
@author: jpzxshi
"""
import numpy as np
import torch
    
class Container(torch.nn.Module):
    def __init__(self, values):
        super(Container, self).__init__()
        for key, value in values.items():
            setattr(self, key, value)

def py2cpp_mesh_f():
    mesh = np.load('./him_data/mesh_482187.npz')
    f = np.load('./him_data/f_482187.npy')
    
    data = {}
    data['points'] = torch.tensor(mesh['points_0'])
    data['vertex'] = torch.tensor(mesh['vertex_0'].astype(np.int64))
    data['line'] = torch.tensor(mesh['line_0'].astype(np.int64))
    data['triangle'] = torch.tensor(mesh['triangle_0'].astype(np.int64))
    data['f'] = torch.tensor(f)
    
    print(data['points'].size(), data['points'].dtype)
    print(data['vertex'].size(), data['vertex'].dtype)
    print(data['line'].size(), data['line'].dtype)
    print(data['triangle'].size(), data['triangle'].dtype)
    print(data['f'].size(), data['f'].dtype)
    
    container = torch.jit.script(Container(data))
    container.save('./him_data/mesh_f_482187.pth')
    
def py2cpp_disc_points_polar():
    points = np.load('./data_random_unit_circle.npy')
    print(points.shape)
    points = np.hstack((points[:, 1:2], points[:, 0:1]))  # r, theta
    print(points.shape)
    print(np.min(points[:, 0]), np.max(points[:, 0]))
    print(np.min(points[:, 1]), np.max(points[:, 1]))
    
    data = {}
    data['disc_points_polar'] = torch.tensor(points)
    
    print(data['disc_points_polar'].size(), data['disc_points_polar'].dtype)
    
    container = torch.jit.script(Container(data))
    container.save('./him_model/sample_points.pth')
    
def py2cpp_model():
    net_py = torch.load('./him_model/model_best.pkl')
    
    r = torch.ones(2, 200, dtype=torch.float64, device=torch.device('cuda'))
    f = torch.ones(2, 5000, dtype=torch.float64, device=torch.device('cuda'))
    x = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float64, device=torch.device('cuda'))
    example = (r, f, x)
    
    #mionet_traced = torch.jit.trace(mionet_py, (example,))
    mionet_traced = torch.jit.optimize_for_inference(torch.jit.trace(net_py, (example,)))
    print(mionet_traced(example))
    
    mionet_traced.save('./him_model/model_best_traced.pt')
    
def py2cpp_mesh_f_u():
    n=1999
    
    mesh = np.load('./mesh_6.npz')
    f = np.load('./data_func_mesh_6.npz')
    u = np.load('./data_solve_mesh_6.npz')
    
    data = {}
    data['points'] = torch.tensor(mesh['points_{}'.format(n)])
    data['vertex'] = torch.tensor(mesh['vertex_{}'.format(n)].astype(np.int64))
    data['line'] = torch.tensor(mesh['line_{}'.format(n)].astype(np.int64))
    data['triangle'] = torch.tensor(mesh['triangle_{}'.format(n)].astype(np.int64))
    data['f'] = torch.tensor(f['func_{}'.format(n)])
    data['u'] = torch.tensor(u['solve_{}'.format(n)])[:, None]
    
    
    print(data['points'].size(), data['points'].dtype)
    print(data['vertex'].size(), data['vertex'].dtype)
    print(data['line'].size(), data['line'].dtype)
    print(data['triangle'].size(), data['triangle'].dtype)
    print(data['f'].size(), data['f'].dtype)
    print(data['u'].size(), data['u'].dtype)
    
    container = torch.jit.script(Container(data))
    container.save('./him_data/mesh_f_u_{}.pth'.format(data['points'].size(0)))
    
def main():
    #py2cpp_mesh_f()
    #py2cpp_disc_points_polar()
    #py2cpp_model()
    
    #py2cpp_mesh_f_u()
    pass

if __name__ == '__main__':
    main()