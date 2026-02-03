import learner as ln

class Poisson_varying_domains_data(ln.data.Data_MIONet_Cartesian):
    '''Data for 2d Poisson equation defined on varying domains.
    '''
    def __init__(self, path):
        super(Poisson_varying_domains_data, self).__init__()
        import numpy as np
        X_train, X_test = np.load(path + '/X_train.npz'), np.load(path + '/X_test.npz')
        self.X_train = (X_train['arr_0'], X_train['arr_1'], X_train['arr_2'], X_train['arr_3'])
        self.y_train = np.load(path + '/y_train.npy')
        self.X_test = (X_test['arr_0'], X_test['arr_1'], X_test['arr_2'] , X_test['arr_3'])
        self.y_test = np.load(path + '/y_test.npy')

def postprocessing(data, net, loss_history):
    #### post processing, for example, plot a figure if needed.
    plot(data, net)
    pass

def plot(data, net):
    import matplotlib.tri as mtri
    import matplotlib.pyplot as plt
    from tools import poly_2_circle
    import numpy as np

    n_train = data.X_train_np[0].shape[0]
    n = 0

    raw_data = np.load('./data/star_raw_data.npz')
    X_test = data.X_test

    domain = X_test[0][n]
    fg = X_test[1][n]
    k = X_test[2][n]
    points = raw_data['points_{}'.format(n_train + n)]
    poly = points[raw_data['vertex_{}'.format(n_train + n)]].reshape(-1,2)
    tri = raw_data['triangle_{}'.format(n_train + n)]
    X = poly_2_circle(points, poly)
    y_true = raw_data['u_{}'.format(n_train + n)]
    y_pred = net.predict((domain, fg, k, X), returnnp=True).reshape(-1)
    # If the given data to be predicted is raw mesh-based data, 
    # it needs to be encoded using the pre-stored 'sample_points.npy' ('sample_points_polar.npy') 
    # as in 'generate_training_data.py'.
    
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    titlesize = 18
    triangle = mtri.Triangulation(points[:, 0], points[:, 1], tri)

    ax[0].set_title('Prediction', size=titlesize)
    tpc0 = ax[0].tripcolor(triangle, y_pred, shading='gouraud', cmap='rainbow')
    fig.colorbar(tpc0, ax=ax[0])

    ax[1].set_title('Ground truth', size=titlesize)
    tpc1 = ax[1].tripcolor(triangle, y_true, shading='gouraud', cmap='rainbow')
    fig.colorbar(tpc1, ax=ax[1])

    ax[2].set_title('Error', size=titlesize)
    tpc2 = ax[2].tripcolor(triangle, np.abs(y_true - y_pred), shading='gouraud', cmap='rainbow')
    fig.colorbar(tpc2, ax=ax[2])

    plt.savefig('prediction_star_d2d.pdf')


def main():
    #### device
    device = 'gpu'  # 'cpu' or 'gpu'
    #### data
    path = 'data/star_d2d/'  # the directory of the dataset
    #### MIONet
    sizes = [
        [200] + [500] * 4 + [1000],
        [5200, -1000], # -1000 means the last layer is without bias
        [5000] + [500] * 4 + [1000],
        [2] + [500] * 4 + [1000],
    ]
    activation = 'relu'
    #### training
    lr = 1e-6
    iterations = 5000000 # sufficiently large
    batch_size = None # None for full batch, using small batch size if out-of-memory error occurs
    print_every = 1000

    training_args = {
        'criterion': 'MSE',
        'optimizer': 'Adam',
        'lr': lr,
        'iterations': iterations,
        'batch_size': batch_size,
        'print_every': print_every,
        'save': 'best_only',
        'callback': None,
        'dtype': 'float',
        'device': device,
    }

    ln.Brain.Start()
    data = Poisson_varying_domains_data(path)
    net = ln.nn.MIONet_Cartesian(sizes, activation, bias=False)
    ln.Brain.Init(data, net)
    ln.Brain.Run(**training_args)
    ln.Brain.Restore()
    ln.Brain.Output(data=False)
    postprocessing(data, ln.Brain.Best_model(), ln.Brain.Loss_history())
    ln.Brain.End()

if __name__ == '__main__':
    main()