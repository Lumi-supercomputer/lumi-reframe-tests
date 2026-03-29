import os
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher
from reframe.utility.osext import cray_cdt_version

class singularity_container_image(rfm.RunOnlyRegressionTest):
    valid_systems       = ['lumi:gpu']
    valid_prog_environs = ['builtin']
    container_platform  = 'Singularity'
    num_gpus_per_node   = 8
    exclusive_access    = True
    #lumi_path_prefix = '/appl/local/containers/easybuild-sif-images'
    laif_path_prefix = '/appl/local/laifs/containers'
    cont_image          = parameter([
        #f'{lumi_path_prefix}/lumi-pytorch-rocm-6.2.1-python-3.12-pytorch-20240918-vllm-4075b35-dockerhash-3cad1babc4b8',
        f'{laif_path_prefix}/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743',
    ])

    num_nodes = parameter([1, 2])

    perf_relative = variable(float, value=0.0, loggable=True)

    @run_before('run')
    def set_launch_settings(self):
        self.env_vars = {
            'NCCL_NET_GDR_LEVEL':'PHB',
            'NCCL_SOCKET_IFNAME':'hsn0,hsn1,hsn2,hsn3',
            'SINGULARITYENV_PREPEND_PATH':'/user-software/bin',
            'SINGULARITYENV_LD_LIBRARY_PATH': '/usr/lib:\$LD_LIBRARY_PATH',
            'MASTER_ADDR': '$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)',
            'MASTER_PORT': '1${SLURM_JOB_ID:0-4}',
        }
        self.job.launcher.options = ['--mpi=pmi2']

@rfm.simple_test
class test_visualtransformer(singularity_container_image):
    launcher = parameter(['torchrun', 'deepspeed'])

    refs = {
        ('torchrun', '1'): (380.0, None, 0.1, 's'),
        ('torchrun', '2'): (220.0, None, 0.1, 's'),
        ('deepspeed', '1'): (380.0, None, 0.1, 's'),
        ('deepspeed', '2'): (220.0, None, 0.1, 's'),
    }

    @run_before('run')
    def configure_nodes(self):
        if self.launcher == 'torchrun':
            self.num_tasks = self.num_nodes
            self.num_tasks_per_node  = 1 
            self.num_cpus_per_task = 56
        else:
            self.num_tasks = self.num_nodes*self.num_gpus_per_node
            self.num_tasks_per_node = self.num_gpus_per_node
            self.num_cpus_per_task = 7
            self.job.launcher.options = ['--mpi=pmi2', '--cpu-bind=v,mask_cpu="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"']

    @run_before('run')
    def set_environment(self):
        self.env_vars['TORCH_HOME'] = '/scratch/project_462000008/.cache'
        self.env_vars['HF_HOME'] = '/scratch/project_462000008/.cache'
        self.env_vars['SINGULARITY_BIND'] = '/appl/local/training/LUMI-AI-Guide/ai-guide-env.sqsh:/user-software:image-src=/,/scratch,/projappl,/project,/flash,/appl,/pfs/lustrep1/scratch/project_462000008,/appl/local/,/pfs/lustrep3/appl/local,/pfs/lustrep2/scratch/project_462000265'
        self.env_vars['OMP_NUM_THREADS'] = 7
        if self.launcher == 'deepspeed':
            self.env_vars['WORLD_SIZE'] = str(self.num_tasks)
            self.env_vars['LOCAL_WORLD_SIZE'] = str(self.num_gpus_per_node)

    @run_before('performance')
    def set_reference(self):
        self.reference = {
            'lumi:gpu': {
                'training_time': self.refs[(self.launcher, str(self.num_nodes))]
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

    @run_after('performance')
    def lower_the_better(self):
        perf_var = 'training_time'
        key_str = self.current_partition.fullname+':'+perf_var
        try:
            found = self.perfvalues[key_str]
        except KeyError:
            return None

        if self.perfvalues[key_str][1] != 0:
            self.perf_relative = ((self.perfvalues[key_str][1]-self.perfvalues[key_str][0])/self.perfvalues[key_str][1])

    @run_before('run')
    def run_training(self):
        self.container_platform.image = f'{self.cont_image}.sif'

        if self.launcher == 'torchrun':
            cmd = 'bash -c "python -m torch.distributed.run --nnodes=\$SLURM_JOB_NUM_NODES --nproc_per_node=8 --rdzv_id=\$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" ddp_visiontransformer.py"'
        elif self.launcher == 'deepspeed':
            cmd = 'bash -c "export RANK=\$SLURM_PROCID; export LOCAL_RANK=\$SLURM_LOCALID; python ds_visiontransformer.py --deepspeed --deepspeed_config ds_config.json"'

        self.container_platform.command = cmd

@rfm.simple_test
class test_container_rccl(singularity_container_image):
    refs = {
        '1': {
            'busbw': (125, -0.05, None, 'GB/s'),
            'algbw': (70, -0.05, None, 'GB/s'),
        },
        '2': {
            'busbw': (85, -0.05, None, 'GB/s'),
            'algbw': (45, -0.05, None, 'GB/s'),
        },
    }

    @run_before('run')
    def configure_nodes(self):
        self.num_tasks_per_node  = 8
        self.num_tasks = self.num_nodes*self.num_tasks_per_node
        self.num_cpus_per_task  = 7
        self.job.launcher.options = ['--mpi=pmi2', '--cpu-bind="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"']

    @run_before('performance')
    def set_reference(self):
        self.reference = {
            'lumi:gpu': self.refs[str(self.num_nodes)]
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

    #@performance_function('GB/s')
    #def algbw(self):
    #    return sn.extractsingle(
    #        r'^\s+134217728.+\s+(?P<algbw>\S+)\s+\S+\s+\S+$',
    #        self.stdout, 'algbw', float
    #    )

    @run_after('performance')
    def higher_the_better(self):
        perf_var = 'busbw'
        key_str = self.current_partition.fullname+':'+perf_var
        try:
            found = self.perfvalues[key_str]
        except KeyError:
            return None

        if self.perfvalues[key_str][1] != 0:
            self.perf_relative = ((self.perfvalues[key_str][0]-self.perfvalues[key_str][1])/self.perfvalues[key_str][1])

    @run_before('run')
    def set_container_variables(self):
        self.container_platform.image = f'{self.cont_image}.sif'
        self.container_platform.command = "bash -c '/usr/libexec/rccl-tests/all_reduce_perf -z 1 -b 2M -e 2048M -f 2 -g 1 -t 1 -R 1 -n 80 -w 5 -d half'"
