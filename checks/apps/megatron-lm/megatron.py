import os
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher
from reframe.utility.osext import cray_cdt_version

class singularity_container_image(rfm.RunOnlyRegressionTest):
    valid_systems       = ['lumi:gpu']
    valid_prog_environs = ['builtin']
    container_platform  = 'Singularity'
    num_tasks_per_node  = 1
    num_gpus_per_node   = 8
    num_cpus_per_task   = 56
    num_tasks           = 2
    exclusive_access    = True
    cont_image          = parameter([
        'rocm-6.2.4-python-3.12-pytorch-v2.6.0',
    ])

    @run_before('run')
    def set_launch_settings(self):
        self.env_vars = {
            'NCCL_NET_GDR_LEVEL':'PHB',
            'NCCL_SOCKET_IFNAME':'hsn0,hsn1,hsn2,hsn3',
            'SINGULARITY_BIND'  :'/var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4,/pfs,/scratch,/projappl,/project,/flash,/appl',
            'MASTER_PORT'       :'29500',
            'MASTER_ADDR'       :'$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)',
        }

@rfm.simple_test
class test_megatron(singularity_container_image):
    sourcesdir = 'https://github.com/ROCm/Megatron-LM'
    reference = {
        'lumi:gpu': {'TFLOPS': (88, -0.05, None, 'TFLOP/s/GPU')}
    }

    @sanity_function
    def check_last_line(self):
        return sn.assert_found(r'throughput per GPU', self.stdout)

    @performance_function('TFLOP/s/GPU')
    def GCD_FLOPS(self):
        return sn.extractsingle(
            r"throughput per GPU:\s+(\d+\.\d+)",
            self.stdout, 1, float
        )

    @run_before('run')
    def run_training(self):
        self.container_platform.image = os.path.join(
            '/appl/local/containers/sif-images/',
            f'lumi-pytorch-{self.cont_image}.sif',
        )

        script_path = os.path.join(os.path.dirname(__file__), 'torch_llama2.sh')
        self.prerun_cmds += [f'cp {script_path} ./']

        self.container_platform.command = 'bash -c "CXX=g++-12 MOCK_DATA=1 USE_FLASH_ATTN=1 GEMM_TUNING=1 TEE_OUTPUT=1 MBS=2 BS=256 TP=2 TE_FP8=0 SEQ_LENGTH=4096 MODEL_SIZE=7 TOTAL_ITERS=4 bash torch_llama2.sh"'
