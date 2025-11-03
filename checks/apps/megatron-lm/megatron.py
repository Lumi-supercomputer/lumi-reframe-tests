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
    exclusive_access    = True
    cont_image          = parameter([
        'rocm-6.2.4-python-3.12-pytorch-v2.6.0',
    ])
    node_config         = parameter(['2node', '8node', '16node'])

    perf_relative = variable(float, value=0.0, loggable=True)

    tags = {'performance'}

    @run_before('run')
    def set_launch_settings(self):
        self.env_vars = {
            'NCCL_NET_GDR_LEVEL':'PHB',
            'NCCL_SOCKET_IFNAME':'hsn0,hsn1,hsn2,hsn3',
            'SINGULARITY_BIND'  :'/var/spool/slurmd,/opt/cray,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4,/pfs,/scratch,/projappl,/project,/flash,/appl',
            'MASTER_PORT'       :'29500',
        }

    @run_before('run')
    def configure_nodes(self):
        if self.node_config   == '2node':
            self.num_tasks = 2
        elif self.node_config == '8node':
            self.num_tasks = 8
        elif self.node_config == '16node':
            self.num_tasks = 16

        self.env_vars['WORLD_SIZE'] = str(self.num_tasks)
        self.env_vars['MASTER_ADDR'] = '$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)'


@rfm.simple_test
class test_megatron(singularity_container_image):
    sourcesdir = 'https://github.com/ROCm/Megatron-LM'

    all_refs = {
        '2node': {
            'GCD TFLOPS': (91, -0.05, None, 'TFLOP/s/GCD'),
            'tokens/GPU/s': (1950, -0.05, None, 'tokens/GPU/s')
        },
        '8node': {
            'GCD TFLOPS': (88, -0.05, None, 'TFLOP/s/GCD'),
            'tokens/GPU/s': (1880, -0.05, None, 'tokens/GPU/s')
        },
        '16node': {
            'GCD TFLOPS': (83, -0.05, None, 'TFLOP/s/GCD'),     #these varied from 74 to 104
            'tokens/GPU/s': (1820, -0.05, None, 'tokens/GPU/s') #these varied similarly from 1700 to 2000
        },
        }


    @run_before('performance')
    def set_reference(self):
        self.reference = {
            'lumi:gpu': self.all_refs[self.node_config]
        }


    @sanity_function
    def check_last_line(self):
        return sn.all([
            sn.assert_found(r'throughput per GPU', self.stdout),
            sn.assert_found(r'tokens/GPU/s', self.stdout)
        ])

    @performance_function('TFLOP/s/GCD')
    def GCD_FLOPS(self):
        return sn.extractsingle(
            r"throughput per GPU:\s+(\d+\.\d+)",
            self.stdout, 1, float
        )
    
    @performance_function('tokens/GPU/s')
    def tokens_per_gpu_per_sec(self):
        return sn.extractsingle(
            r"tokens/GPU/s:\s+(\d+\.\d+)",
            self.stdout, 1, float
        )

    @run_after('performance')
    def higher_the_better(self):
        perf_var = 'tokens_per_gpu_per_sec'
        key_str = self.current_partition.fullname+':'+perf_var
        try:
            found = self.perfvalues[key_str]
        except KeyError:
            return None

        if self.perfvalues[key_str][1] != 0:
            self.perf_relative = ((self.perfvalues[key_str][0]-self.perfvalues[key_str][1])/self.perfvalues[key_str][1])

    @run_before('run')
    def run_training(self):
        self.container_platform.image = os.path.join(
            '/appl/local/containers/sif-images/',
            f'lumi-pytorch-{self.cont_image}.sif',
        )

        script_path = os.path.join(os.path.dirname(__file__), 'torch_llama2.sh')
        self.prerun_cmds += [f'cp {script_path} ./']

        self.container_platform.command = 'bash -c "CXX=g++-12 MOCK_DATA=0 USE_FLASH_ATTN=1 GEMM_TUNING=1 TEE_OUTPUT=1 MBS=2 BS=256 TP=2 TE_FP8=0 SEQ_LENGTH=4096 MODEL_SIZE=7 TOTAL_ITERS=10 DATA_PATH=/project/project_462000008/datasets/megatron/data_text_document DATA_CACHE_PATH=/project/project_462000008/datasets/megatron/.cache bash torch_llama2.sh"'
