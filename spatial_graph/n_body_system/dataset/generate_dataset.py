import pickle as pkl
import numpy as np
import argparse
from system import System
from tqdm import tqdm
import os
from joblib import Parallel, delayed


parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='charged',
                    help='What simulation to generate.')
parser.add_argument('--path', type=str, default='data',
                    help='Path to save.')
parser.add_argument('--num-train', type=int, default=10000,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-valid', type=int, default=2000,
                    help='Number of validation simulations to generate.')
parser.add_argument('--num-test', type=int, default=2000,
                    help='Number of test simulations to generate.')
parser.add_argument('--length', type=int, default=5000,
                    help='Length of trajectory.')
parser.add_argument('--length_test', type=int, default=5000,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--n_isolated', type=int, default=5,
                    help='Number of isolated balls in the simulation.')
parser.add_argument('--n_stick', type=int, default=0,
                    help='Number of sticks in the simulation.')
parser.add_argument('--n_hinge', type=int, default=0,
                    help='Number of hinges in the simulation.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--suffix', type=str, default="",
                    help='add a suffix to the name')
parser.add_argument('--n_workers', type=int, default=1,
                    help="Number of workers")
parser.add_argument('--box_size', type=float, default=None,
                    help="The size of the box.")

args = parser.parse_args()

suffix = '_charged'

suffix += str(args.n_isolated) + '_' + str(args.n_stick) + '_' + str(args.n_hinge) + args.suffix
np.random.seed(args.seed)

print(suffix)


def para_comp(length, sample_freq):
    while True:
        X, V = [], []
        # X, V = np.empty(length // sample_freq, dtype=object), np.empty(length // sample_freq, dtype=object)
        system = System(n_isolated=args.n_isolated, n_stick=args.n_stick, n_hinge=args.n_hinge, box_size=args.box_size)
        for t in range(length):
            system.simulate_one_step()
            if t % sample_freq == 0:
                X.append(system.X.copy())
                V.append(system.V.copy())
                # X[t // sample_freq] = system.X.copy()
                # V[t // sample_freq] = system.V.copy()
        system.check()
        assert system.is_valid()  # currently do not apply constraint
        if system.is_valid():
            cfg = system.configuration()
            X = np.array(X)
            V = np.array(V)
            return cfg, X, V, system.edges, system.charges


def generate_dataset(num_sims, length, sample_freq):
    results = Parallel(n_jobs=args.n_workers)(delayed(para_comp)(length, sample_freq) for i in tqdm(range(num_sims)))
    cfg_all, loc_all, vel_all, edges_all, charges_all = zip(*results)
    # print(f'total trials: {cnt:d}, samples: {len(loc_all):d}', cnt)

    return loc_all, vel_all, edges_all, charges_all, cfg_all


if __name__ == "__main__":
    if not os.path.exists(args.path):
        os.mkdir(args.path)

    print("Generating {} training simulations".format(args.num_train))
    loc_train, vel_train, edges_train, charges_train, cfg_train = generate_dataset(args.num_train,
                                                                                      args.length,
                                                                                      args.sample_freq)
    np.save(os.path.join(args.path, 'loc_train' + suffix + '.npy'), loc_train)
    np.save(os.path.join(args.path, 'vel_train' + suffix + '.npy'), vel_train)
    np.save(os.path.join(args.path, 'edges_train' + suffix + '.npy'), edges_train)
    np.save(os.path.join(args.path, 'charges_train' + suffix + '.npy'), charges_train)
    with open(os.path.join(args.path, 'cfg_train' + suffix + '.pkl'), 'wb') as f:
        pkl.dump(cfg_train, f)

    print("Generating {} validation simulations".format(args.num_valid))
    loc_valid, vel_valid, edges_valid, charges_valid, cfg_valid = generate_dataset(args.num_valid,
                                                                                      args.length,
                                                                                      args.sample_freq)
    np.save(os.path.join(args.path, 'loc_valid' + suffix + '.npy'), loc_valid)
    np.save(os.path.join(args.path, 'vel_valid' + suffix + '.npy'), vel_valid)
    np.save(os.path.join(args.path, 'edges_valid' + suffix + '.npy'), edges_valid)
    np.save(os.path.join(args.path, 'charges_valid' + suffix + '.npy'), charges_valid)
    with open(os.path.join(args.path, 'cfg_valid' + suffix + '.pkl'), 'wb') as f:
        pkl.dump(cfg_valid, f)

    print("Generating {} test simulations".format(args.num_test))
    loc_test, vel_test, edges_test, charges_test, cfg_test = generate_dataset(args.num_test,
                                                                                 args.length_test,
                                                                                 args.sample_freq)
    np.save(os.path.join(args.path, 'loc_test' + suffix + '.npy'), loc_test)
    np.save(os.path.join(args.path, 'vel_test' + suffix + '.npy'), vel_test)
    np.save(os.path.join(args.path, 'edges_test' + suffix + '.npy'), edges_test)
    np.save(os.path.join(args.path, 'charges_test' + suffix + '.npy'), charges_test)
    with open(os.path.join(args.path, 'cfg_test' + suffix + '.pkl'), 'wb') as f:
        pkl.dump(cfg_test, f)

# python -u generate_dataset.py --num-train 5000 --seed 43 --n_isolated 0 --n_stick 5 --n_hinge 0  --n_workers 50
