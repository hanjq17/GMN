import argparse
import torch
import torch.utils.data
from motion.dataset import MotionDataset
from n_body_system.model import GNN, Baseline, Linear, EGNN_vel, Linear_dynamics, RF_vel, GMN
import os
from torch import nn, optim
import json

import random
import numpy as np

parser = argparse.ArgumentParser(description='Graph Mechanics Networks')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='n_body_system/logs', metavar='N',
                    help='folder to output the json log file')
parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=64, metavar='N',
                    help='hidden dim')
parser.add_argument('--model', type=str, default='egnn_vel', metavar='N',
                    help='available models: gnn, baseline, linear, linear_vel, egnn_vel, rf_vel')
parser.add_argument('--attention', type=int, default=0, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--weight_decay', type=float, default=1e-12, metavar='N',
                    help='timing experiment')
parser.add_argument('--norm_diff', type=eval, default=False, metavar='N',
                    help='normalize_diff')
parser.add_argument('--tanh', type=eval, default=False, metavar='N',
                    help='use tanh')
parser.add_argument('--delta_frame', type=int, default=50,
                    help='Number of frames delta.')
parser.add_argument('--data_dir', type=str, default='spatial_graph/motion',
                    help='Data directory.')
parser.add_argument('--learnable', type=eval, default=False, metavar='N',
                    help='Use learnable FK.')

parser.add_argument("--config_by_file", default=False, action="store_true", )


args = parser.parse_args()
if args.config_by_file:
    job_param_path = 'configs/simple_config_motion.json'
    with open(job_param_path, 'r') as f:
        hyper_params = json.load(f)
        args.exp_name = hyper_params["exp_name"]
        args.batch_size = hyper_params["batch_size"]
        args.epochs = hyper_params["epochs"]
        args.no_cuda = hyper_params["no_cuda"]
        args.seed = hyper_params["seed"]
        args.lr = hyper_params["lr"]
        args.nf = hyper_params["nf"]
        args.model = hyper_params["model"]
        args.attention = hyper_params["attention"]
        args.n_layers = hyper_params["n_layers"]
        args.max_training_samples = hyper_params["max_training_samples"]
        args.data_dir = hyper_params["data_dir"]
        args.weight_decay = hyper_params["weight_decay"]
        args.norm_diff = hyper_params["norm_diff"]
        args.tanh = hyper_params["tanh"]
        args.learnable = hyper_params["learnable"]

        args.delta_frame = hyper_params["delta_frame"]

args.cuda = not args.no_cuda and torch.cuda.is_available()


device = torch.device("cuda" if args.cuda else "cpu")
loss_mse = nn.MSELoss()

print(args)
try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs(args.outf + "/" + args.exp_name)
except OSError:
    pass

# torch.autograd.set_detect_anomaly(True)


def get_velocity_attr(loc, vel, rows, cols):

    diff = loc[cols] - loc[rows]
    norm = torch.norm(diff, p=2, dim=1).unsqueeze(1)
    u = diff/norm
    va, vb = vel[rows] * u, vel[cols] * u
    va, vb = torch.sum(va, dim=1).unsqueeze(1), torch.sum(vb, dim=1).unsqueeze(1)
    return va


def main():
    # fix seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset_train = MotionDataset(partition='train', max_samples=args.max_training_samples, data_dir=args.data_dir,
                                  delta_frame=args.delta_frame)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                               num_workers=8)

    dataset_val = MotionDataset(partition='val', max_samples=600, data_dir=args.data_dir,
                                  delta_frame=args.delta_frame)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                             num_workers=8)

    dataset_test = MotionDataset(partition='test', max_samples=600, data_dir=args.data_dir,
                                  delta_frame=args.delta_frame)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                              num_workers=8)

    if args.model == 'gnn':
        model = GNN(input_dim=6, hidden_nf=args.nf, n_layers=args.n_layers, device=device, recurrent=True)
    elif args.model == 'egnn_vel':
        model = EGNN_vel(in_node_nf=2, in_edge_nf=2 + 1, hidden_nf=args.nf, device=device, n_layers=args.n_layers,
                         recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)
    elif args.model == 'egnn_vel_cons':
        model = EGNN_vel(in_node_nf=2, in_edge_nf=2 + 1, hidden_nf=args.nf, device=device, n_layers=args.n_layers,
                         recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)
    elif args.model == 'gmn':
        model = GMN(in_node_nf=2, in_edge_nf=2 + 1, hidden_nf=args.nf, device=device, n_layers=args.n_layers,
                    recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh, learnable=args.learnable)
    elif args.model == 'baseline':
        model = Baseline()
    elif args.model == 'linear_vel':
        model = Linear_dynamics(device=device)
    elif args.model == 'linear':
        model = Linear(6, 3, device=device)
    elif args.model == 'rf_vel':
        model = RF_vel(hidden_nf=args.nf, edge_attr_nf=2 + 3, device=device, act_fn=nn.SiLU(), n_layers=args.n_layers)
    else:
        raise Exception("Wrong model specified")

    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    results = {'epochs': [], 'loss': [], 'train loss': []}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    best_train_loss = 1e8
    for epoch in range(args.epochs):
        train_loss = train(model, optimizer, epoch, loader_train)
        results['train loss'].append(train_loss)
        if epoch % args.test_interval == 0:
            val_loss = train(model, optimizer, epoch, loader_val, backprop=False)
            test_loss = train(model, optimizer, epoch, loader_test, backprop=False)
            results['epochs'].append(epoch)
            results['loss'].append(test_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_train_loss = train_loss
                best_epoch = epoch
                # torch.save(model.state_dict(), args.outf + '/' + 'saved_model.pth')
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best apoch %d"
                  % (best_val_loss, best_test_loss, best_epoch))

        json_object = json.dumps(results, indent=4)
        with open(args.outf + "/" + args.exp_name + "/loss.json", "w") as outfile:
            outfile.write(json_object)
    return best_train_loss, best_val_loss, best_test_loss, best_epoch


def train(model, optimizer, epoch, loader, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0, 'loss_stick': 0, 'loss_vel': 0, 'reg_loss': 0}
    # res_energy = {'gt': 0, 'method': 0}

    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, _ = data[0].size()
        data, cfg = data[:-1], data[-1]
        data = [d.to(device) for d in data]
        data = [d.view(-1, d.size(2)) for d in data]  # construct mini-batch graphs
        loc, vel, edge_attr, charges, loc_end, vel_end, Z = data

        edges = loader.dataset.get_edges(batch_size, n_nodes)
        edges = [edges[0].to(device), edges[1].to(device)]

        cfg = loader.dataset.get_cfg(batch_size, n_nodes, cfg)
        cfg = {_: cfg[_].to(device) for _ in cfg}

        optimizer.zero_grad()

        # helper to compute reg loss
        reg_loss = 0

        if 'Stick' in cfg:
            stick = cfg['Stick']
            id0, id1 = stick[..., 0], stick[..., 1]
            stick_len_input = torch.sqrt(torch.sum((loc[id0] - loc[id1]) ** 2, dim=-1))
        if 'Hinge' in cfg:
            hinge = cfg['Hinge']
            id0, id1, id2 = hinge[..., 0], hinge[..., 1], hinge[..., 2]
            stick_len_input1 = torch.sqrt(torch.sum((loc[id0] - loc[id1]) ** 2, dim=-1))
            stick_len_input2 = torch.sqrt(torch.sum((loc[id0] - loc[id2]) ** 2, dim=-1))

        if args.model == 'gnn':
            nodes = torch.cat([loc, vel], dim=1)
            loc_pred = model(nodes, edges, edge_attr)
        elif args.model == 'egnn_vel' or args.model == 'egnn_vel_cons':
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            nodes = torch.cat((nodes, Z / Z.max()), dim=-1)
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
            loc_pred = model(nodes, loc.detach(), edges, vel, edge_attr)
        elif args.model == 'gmn':
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            nodes = torch.cat((nodes, Z / Z.max()), dim=-1)
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
            loc_pred, vel_pred = model(nodes, loc.detach(), edges, vel, cfg, edge_attr)

        elif args.model == 'baseline':
            backprop = False
            loc_pred = model(loc)
        elif args.model == 'linear':
            loc_pred = model(torch.cat([loc, vel], dim=1))
        elif args.model == 'linear_vel':
            loc_pred = model(loc, vel)
        elif args.model == 'rf_vel':
            rows, cols = edges
            vel_norm = torch.sqrt(torch.sum(vel ** 2, dim=1).unsqueeze(1)).detach()
            loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, 1).unsqueeze(1)
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()
            loc_pred = model(vel_norm, loc.detach(), edges, vel, edge_attr)
        else:
            raise Exception("Wrong model")

        # compute regularization loss

        if 'Stick' in cfg:
            stick = cfg['Stick']
            id0, id1 = stick[..., 0], stick[..., 1]
            stick_len_output = torch.sqrt(torch.sum((loc_pred[id0] - loc_pred[id1]) ** 2, dim=-1))
            reg_loss = reg_loss + torch.mean(torch.abs(stick_len_input - stick_len_output))
        if 'Hinge' in cfg:
            hinge = cfg['Hinge']
            id0, id1, id2 = hinge[..., 0], hinge[..., 1], hinge[..., 2]
            stick_len_output1 = torch.sqrt(torch.sum((loc_pred[id0] - loc_pred[id1]) ** 2, dim=-1))
            stick_len_output2 = torch.sqrt(torch.sum((loc_pred[id0] - loc_pred[id2]) ** 2, dim=-1))
            reg_loss = reg_loss + torch.mean(torch.abs(stick_len_input1 - stick_len_output1)) + torch.mean(
                torch.abs(stick_len_input2 - stick_len_output2))

        loss = loss_mse(loc_pred, loc_end)

        if backprop:
            if args.model == 'egnn_vel_cons':
                (loss + 0.1 * reg_loss).backward()
            else:
                loss.backward()
            optimizer.step()
        res['loss'] += loss.item()*batch_size
        try:
            res['reg_loss'] += reg_loss.item()*batch_size
        except:  # no reg loss (no sticks and hinges)
            pass
        res['counter'] += batch_size

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f reg loss: %.5f'
          % (prefix+loader.dataset.partition, epoch,
             res['loss'] / res['counter'], res['reg_loss'] / res['counter']))

    return res['loss'] / res['counter']


if __name__ == "__main__":
    best_train_loss, best_val_loss, best_test_loss, best_epoch = main()
    print("best_train = %.6f" % best_train_loss)
    print("best_val = %.6f" % best_val_loss)
    print("best_test = %.6f" % best_test_loss)
    print("best_epoch = %d" % best_epoch)





