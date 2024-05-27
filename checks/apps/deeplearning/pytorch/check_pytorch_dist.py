import os
import reframe as rfm
import reframe.utility.sanity as sn


class pytorch_distr_cnn_base(rfm.RunOnlyRegressionTest):
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin']
    sourcesdir = 'src'
    exclusive_access = True
    num_tasks = 16
    num_tasks_per_node = 8
    num_gpus_per_node = 8
    env_vars = {
        'NCCL_DEBUG': 'INFO',
        'NCCL_SOCKET_IFNAME': 'hsn0,hsn1,hsn2,hsn3',
        'NCCL_NET_GDR_LEVEL': '2', # Fails to work on multiple nodes with NCCL_NET_GDR_LEVEL=3
        'MIOPEN_USER_DB_PATH': '/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}',
        'MIOPEN_CUSTOM_CACHE_DIR': '${MIOPEN_USER_DB_PATH}'
    }
    throughput_per_gpu = 193.98
    throughput_total = throughput_per_gpu * num_tasks
    reference = {
        'lumi:gpu': {
            'samples_per_sec_per_gpu': (throughput_per_gpu,
                                        -0.1, None, 'samples/sec'),
            'samples_per_sec_total': (throughput_total,
                                      -0.1, None, 'samples/sec')
        }
    }

    @run_before('run')
    def set_cpu_binding(self):
        self.job.launcher.options = ['--cpu-bind="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"']

    @sanity_function
    def assert_job_is_complete(self):
        return sn.all([
            sn.assert_found(r'Using network AWS Libfabric', self.stdout),
            sn.assert_found(r'Selected Provider is cxi', self.stdout),
            sn.assert_found(r'Total average', self.stdout)  
        ])

    @performance_function('samples/sec')
    def samples_per_sec_per_gpu(self):
        return sn.avg(sn.extractall(
            r'Epoch\s+\d+\:\s+(?P<samples_per_sec_per_gpu>\S+)\s+images',
            self.stdout, 'samples_per_sec_per_gpu', float
        ))

    @performance_function('samples/sec')
    def samples_per_sec_total(self):
        return sn.avg(sn.extractall(
            r'Total average: (?P<samples_per_sec_total>\S+)\s+images',
            self.stdout, 'samples_per_sec_total', float
        ))


@rfm.simple_test
class pytorch_distr_cnn_container_module(pytorch_distr_cnn_base):
    modules = ['PyTorch']

    tags = {'python', 'contrib'}

    @run_before('run')
    def set_container_variables(self):
        self.container_platform = 'Singularity'
        self.container_platform.image = '$SIFPYTORCH'
        self.container_platform.command = 'conda-python-distributed -u cnn_distr.py --gpu --modelpath model'


@rfm.simple_test
class pytorch_distr_cnn_container_direct(pytorch_distr_cnn_base):

    tags = {'singularity', 'python'}

    @run_before('run')
    def set_container_variables(self):
        self.container_platform = 'Singularity'
        self.container_platform.image = os.path.join(
            '/appl/local/containers',
            'sif-images',
            'lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.1.0.sif'
        )
        self.container_platform.command = 'bash conda-python-distributed.sh -u cnn_distr.py --gpu --modelpath model'

        self.container_platform.mount_points = [
            ('/var/spool/slurmd', '/var/spool/slurmd'),
            ('/opt/cray', '/opt/cray'),
            ('/usr/lib64/libcxi.so.1', '/usr/lib64/libcxi.so.1'),
            ('/usr/lib64/libjansson.so.4', '/usr/lib64/libjansson.so.4'),
        ]
