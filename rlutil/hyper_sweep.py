"""
Usage

args = {
'param1': [1e-3, 1e-2, 1e-2],
'param2': [1,5,10,20],
}

run_sweep_parallel(func, args)

or

run_sweep_serial(func, args)

"""
import os
import itertools
import multiprocessing
import random
import hashlib
from datetime import datetime


def _recurse_helper(d):
    return itertools.product(
        *[_recurse_helper(v) if isinstance(v, dict) else v for v in d.values()]
        )

def _map_to_kwargs(d, config):
    new_d = d.copy()
    for k, c in zip(d.keys(), config):
        if isinstance(d[k], dict):
            new_d[k] = _map_to_kwargs(d[k], c)
        else:
            new_d[k] = c
    return new_d

def fixed_config(hyper_config):
    new_config = hyper_config.copy()
    for k in new_config:
        if isinstance(new_config[k], dict):
            new_config[k] = fixed_config(new_config[k])
        else:
            new_config[k] = (new_config[k],)
    return new_config

class Sweeper(object):
    def __init__(self, hyper_config, repeat):
        self.hyper_config = hyper_config
        self.repeat = repeat

    def __iter__(self):
        count = 0
        for _ in range(self.repeat):
            for config in _recurse_helper(self.hyper_config):
                kwargs = _map_to_kwargs(self.hyper_config, config)
                count += 1
                yield kwargs


def always_true(x):
    return True


def chunk_filter(chunk_id, num_chunks):
    def filter(config):
        hash_ = int(hashlib.md5(repr(config).encode('utf-8')).hexdigest(), 16)
        task_chunk = hash_ % num_chunks
        return chunk_id == task_chunk
    return filter


def run_sweep_serial(run_method, params, repeat=1, filter_fn=always_true):
    sweeper = Sweeper(params, repeat)
    for config in sweeper:
        if filter_fn(config):
            run_method(**config)


def kwargs_wrapper(args_method_seed):
    args, method, seed = args_method_seed
    from rlutil import seeding
    seeding.set_seed(seed)
    return method(**args)


def run_sweep_parallel(run_method, params, repeat=1, num_cpu=multiprocessing.cpu_count(), 
    filter_fn=always_true):
    sweeper = Sweeper(params, repeat)
    pool = multiprocessing.Pool(num_cpu)
    exp_args = []
    exp_n = 0
    for config in sweeper:
        if filter_fn(config):
            exp_args.append((config, run_method, exp_n))
            exp_n += 1
    print('Launching {exp_n} experiments with {num_cpu} CPUs'.format(**locals()))
    random.shuffle(exp_args)
    pool.map(kwargs_wrapper, exp_args)


THIS_FILE_DIR = os.path.dirname(__file__)
SCRIPTS_DIR = os.path.join(os.path.dirname(THIS_FILE_DIR), 'scripts')
def run_sweep_doodad(run_method, params, run_mode, mounts, repeat=1, test_one=False):
    import doodad
    sweeper = Sweeper(params, repeat)
    for config in sweeper:
        def run_method_args():
            run_method(**config)
        doodad.launch_python(
                target = os.path.join(SCRIPTS_DIR, 'run_experiment_lite_doodad.py'),
                mode=run_mode,
                mount_points=mounts,
                use_cloudpickle=True,
                args = {'run_method': run_method_args},
        )
        if test_one:
            break


if __name__ == "__main__":
    def example_run_method(exp_name, param1, param2='a', param3=3, param4=4):
        import time
        time.sleep(1.0)
        print(exp_name, param1, param2, param3, param4)
    sweep_op = {
        'param1': [1e-3, 1e-2, 1e-1],
        'param2': [1,5,10,20],
        'param3': [True, False]
    }
    run_sweep_parallel(example_run_method, sweep_op, repeat=2)
