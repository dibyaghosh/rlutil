import os

import doodad as pd
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.easy_sweep import hyper_sweep
import os.path as osp
import glob

instance_types = {
    'c4.large': dict(instance_type='c4.large',spot_price=0.20),
    'c4.xlarge': dict(instance_type='c4.xlarge',spot_price=0.20),
    'c4.2xlarge': dict(instance_type='c4.2xlarge',spot_price=0.50),
    'c4.4xlarge': dict(instance_type='c4.4xlarge',spot_price=0.50),
    'c5.large': dict(instance_type='c5.large',spot_price=0.20),
    'c5.xlarge': dict(instance_type='c5.xlarge',spot_price=0.20),
    'c5.2xlarge': dict(instance_type='c5.2xlarge',spot_price=0.50),
    'c5.4xlarge': dict(instance_type='c5.4xlarge',spot_price=0.50),
}

def launch(project_dir, method, params, mode='local', code_dependencies=list(), data_dependencies=dict(), instance_type='c4.xlarge', docker_image='justinfu/rlkit:0.4', s3_log_prefix='rlkit'):
    """
    Arguments:
        project_dir (str): The main directory containing all project files. Data will be saved to project_dir/data
            This will additionally add all dependencies in project_dir/dependencies
        method (fn): The function to call
        params (dict): The set of hyperparameters to sweep over
        mode (str): Choose between ['ec2', 'docker', 'local']
        code_dependencies (list): List of code locations that need to be additionally mounted in the Docker image and added to the python path.
        data_dependencies (dict): for remote_location:data_dir in this dict, the directory `data_dir` will be accessible in the Docker image at `/tmp/data/{remote_location}`
        instance_type (str / dict): if str, uses an instance type from `instance_types` above. If dict, need keys `instance_type` and `spot_price`
        docker_image (str): Name of docker image
        s3_log_prefix (str): Where data will be stored on s3
    """

    PROJECT_DIR = osp.realpath(project_dir)
    LOCAL_OUTPUT_DIR = osp.join(PROJECT_DIR,'data')

    # Set up code and output directories
    REMOTE_OUTPUT_DIR = '/tmp/outputs'  # this is the directory visible to the target
    REMOTE_DATA_DIR = '/tmp/data'
    main_mount = mount.MountLocal(local_dir=PROJECT_DIR, pythonpath=True, filter_dir=('data','analysis','dependencies'))

    all_code_dependencies = [
        item
        for item in glob.glob(osp.join(PROJECT_DIR, "dependencies/*")) + code_dependencies
        if osp.isdir(item)
    ]

    code_mounts = [main_mount] + [mount.MountLocal(local_dir=directory, pythonpath=True) for directory in all_code_dependencies]

    # MuJoCo
    code_mounts.append(mount.MountLocal(local_dir=osp.expanduser('~/.mujoco'), mount_point='/root/.mujoco', pythonpath=True))
    code_mounts.append(mount.MountLocal(local_dir=osp.expanduser('~/projects/rank_collapse/doodad_old/'), pythonpath=True))

    params['output_dir'] = [REMOTE_OUTPUT_DIR]
    params['data_dir'] = [REMOTE_DATA_DIR]

    if mode == 'local':
        doodad_mode = pd.mode.Local()
        params['output_dir'] = [LOCAL_OUTPUT_DIR]
    elif mode == 'docker':
        doodad_mode = pd.mode.LocalDocker(
            image=docker_image
        )
    elif mode == 'ec2':
        assert instance_type in instance_types
        doodad_mode = pd.mode.EC2AutoconfigDocker(
            image=docker_image,
            image_id='ami-086ecbff428fa44a8',
            region='us-west-1',  # EC2 region
            s3_log_prefix=s3_log_prefix, # Folder to store log files under
            s3_log_name=s3_log_prefix,
            terminate=True,  # Whether to terminate on finishing job
            **instance_types[instance_type]
        )

    data_mounts = [
        mount.MountLocal(local_dir=osp.realpath(directory), mount_point=osp.join(REMOTE_DATA_DIR,remote_name))
        for remote_name,directory in data_dependencies.items()
    ]

    if mode == 'local':
        output_mounts = []
    elif mode == 'docker' or mode == 'ssh':
        output_dir = osp.join(LOCAL_OUTPUT_DIR, 'docker/')
        output_mounts= [mount.MountLocal(local_dir=output_dir, mount_point=REMOTE_OUTPUT_DIR,output=True)]
    elif mode == 'ec2':
        output_mounts = [mount.MountS3(s3_path='data',mount_point=REMOTE_OUTPUT_DIR,output=True)]


    mounts = code_mounts + data_mounts + output_mounts

    hyper_sweep.run_sweep_doodad(method, params, doodad_mode, mounts)
