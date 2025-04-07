import os
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher
from reframe.utility.osext import cray_cdt_version

class singularity_container_image(rfm.RunOnlyRegressionTest):
    valid_systems = ['lumi:gpu']
    container_platform  = 'Singularity'
    num_tasks           = 8
    num_tasks_per_node  = 8
    num_gpus_per_node   = 8
    cpus_per_task       = 7
    exclusive_access    = True

    @run_before('run')
    def set_launch_settings(self):
        self.env_vars = {
            'NCCL_NET_GDR_LEVEL':'PHB',
            'NCCL_SOCKET_IFNAME':'hsn0,hsn1,hsn2,hsn3',
            'SINGULARITY_BIND'  :'/appl/local/training/LUMI-AI-Guide/visualtransformer-env.sqsh:/user-software:image-src=/,/var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4,/pfs,/scratch,/projappl,/project,/flash,/appl',
            'MASTER_ADDR'       :'$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)',
            'MASTER_PORT'       :'29500',
            'WORLD_SIZE'        :'8',
            'LOCAL_WORLD_SIZE'  :'8',
        }
        self.job.launcher.options = ['--cpu-bind=v,mask_cpu="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"']

@rfm.simple_test
class test_pytorch_container(singularity_container_image):
    valid_prog_environs = ['builtin']
    cont_image = parameter([
        'rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35',
    ])
    reference = {
            'lumi:gpu': {
                'training_time' : (380.00, None, 0.1, 's'),
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
    def set_container_variables(self):
        self.container_platform.image = os.path.join(
                '/appl/local/containers/sif-images/',
                f'lumi-pytorch-{self.cont_image}.sif',
                )
        self.container_platform.command = 'bash -c "\$WITH_CONDA; export RANK=\$SLURM_PROCID && export LOCAL_RANK=\$SLURM_LOCALID && python ddp_visualtransformer.py"'

@rfm.simple_test
class test_container_rccl(singularity_container_image):
    valid_prog_environs = ['builtin']
    reference = {
        'lumi:gpu': {
            'busbw': (125, -0.05, None, 'GB/s'),
            'algbw': (70, -0.05, None, 'GB/s'),
        }
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
                'lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35.sif'
                )
        self.container_platform.command = "bash -c '\$WITH_CONDA; /opt/rccltests/all_reduce_perf -z 1 -b 2M -e 2048M -f 2 -g 1 -t 1 -R 1 -n 80 -w 5 -d half'"
