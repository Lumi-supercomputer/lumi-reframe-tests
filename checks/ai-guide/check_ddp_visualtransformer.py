import os
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher
from reframe.utility.osext import cray_cdt_version

class singularity_container_image(rfm.RunOnlyRegressionTest):
    valid_systems       = ['lumi:gpu']
    valid_prog_environs = ['builtin']
    container_platform  = 'Singularity'
    num_tasks_per_node  = 8
    num_gpus_per_node   = 8
    num_cpus_per_task   = 7
    exclusive_access    = True
    cont_image          = parameter([
        'rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35',
    ])
    node_config = parameter(['1node', '2node'])

    @run_before('run')
    def set_launch_settings(self):
        self.env_vars = {
            'TORCH_HOME':'/scratch/project_462000008/.cache',
            'HF_HOME':'/scratch/project_462000008/.cache',
            'NCCL_NET_GDR_LEVEL':'PHB',
            'NCCL_SOCKET_IFNAME':'hsn0,hsn1,hsn2,hsn3',
            'SINGULARITY_BIND'  :'/appl/local/training/LUMI-AI-Guide/visualtransformer-env.sqsh:/user-software:image-src=/,/var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4,/pfs,/scratch,/projappl,/project,/flash,/appl,/appl/local/training/LUMI-AI-Guide/deepspeed_adam:/user-software/lib/python3.12/site-packages/deepspeed/ops/csrc/adam,/appl/local/training/LUMI-AI-Guide/deepspeed_includes:/user-software/lib/python3.12/site-packages/deepspeed/ops/csrc/includes',
            'SINGULARITYENV_PREPEND_PATH':'/user-software/bin',
            'MASTER_PORT'       :'29500',
            'LOCAL_WORLD_SIZE'  :'8',
        }
        self.job.launcher.options = ['--cpu-bind=v,mask_cpu="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"']

    @run_before('run')
    def configure_nodes(self):
        if self.node_config == '1node':
            self.num_tasks = 8
        elif self.node_config == '2node':
            self.num_tasks = 16

        self.env_vars['WORLD_SIZE'] = str(self.num_tasks)
        self.env_vars['MASTER_ADDR'] = '$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)'

@rfm.simple_test
class test_visualtransformer(singularity_container_image):
    launcher = parameter(['pytorch', 'deepspeed'])

    refs = {
        ('pytorch', '1node'): (380.0, None, 0.1, 's'),
        ('pytorch', '2node'): (220.0, None, 0.1, 's'),
        ('deepspeed', '1node'): (380.0, None, 0.1, 's'),
        ('deepspeed', '2node'): (220.0, None, 0.1, 's'),
    }


    @run_before('performance')
    def set_reference(self):
        self.reference = {
            'lumi:gpu': {
                'training_time': self.refs[(self.launcher, self.node_config)]
            }
        }

    @sanity_function
    def check_last_line(self):
        return sn.assert_found(r'Time elapsed', self.stdout)

    @performance_function('s')
    def training_time(self):
        return sn.extractsingle(
            r"Time elapsed \(s\): (\d+\.\d+)",
            self.stdout, 1, float
        )

    @run_before('run')
    def run_training(self):
        self.container_platform.image = os.path.join(
            '/appl/local/containers/sif-images/',
            f'lumi-pytorch-{self.cont_image}.sif',
        )

        base = (
            'bash -c "export RANK=\$SLURM_PROCID; export LOCAL_RANK=\$SLURM_LOCALID;'
        )

        if self.launcher == 'pytorch':
            cmd = 'python ddp_visualtransformer.py'
        elif self.launcher == 'deepspeed':
            cmd = 'export CXX=g++-12; python ds_visualtransformer.py --deepspeed --deepspeed_config ds_config.json'

        self.container_platform.command = f'{base}{cmd}"'

@rfm.simple_test
class test_container_rccl(singularity_container_image):
    refs = {
        '1node': {
            'busbw': (125, -0.05, None, 'GB/s'),
            'algbw': (70, -0.05, None, 'GB/s'),
        },
        '2node': {
            'busbw': (85, -0.05, None, 'GB/s'),
            'algbw': (45, -0.05, None, 'GB/s'),
        },
    }

    @run_before('performance')
    def set_reference(self):
        self.reference = {
            'lumi:gpu': self.refs[self.node_config]
        }

    @sanity_function
    def check_last_line(self):
        return sn.assert_found(r'Avg bus bandwidth', self.stdout)

    @performance_function('GB/s')
    def busbw(self):
        return sn.extractsingle(
            r'^\s+134217728.+\s+(?P<busbw>\S+)\s+\S+$',
            self.stdout, 'busbw', float
        )

    @performance_function('GB/s')
    def algbw(self):
        return sn.extractsingle(
            r'^\s+134217728.+\s+(?P<algbw>\S+)\s+\S+\s+\S+$',
            self.stdout, 'algbw', float
        )

    @run_before('run')
    def set_container_variables(self):
        self.container_platform.image = os.path.join(
                '/appl/local/containers/sif-images/',
                f'lumi-pytorch-{self.cont_image}.sif',
                )
        self.container_platform.command = "bash -c '/opt/rccltests/all_reduce_perf -z 1 -b 2M -e 2048M -f 2 -g 1 -t 1 -R 1 -n 80 -w 5 -d half'"
