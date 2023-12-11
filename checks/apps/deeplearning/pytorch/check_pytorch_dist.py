import os
import reframe as rfm
import reframe.utility.sanity as sn


class pytorch_distr_cnn_base(rfm.RunOnlyRegressionTest):
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin']
    sourcesdir = 'src'
    exclusive_access = True
    num_tasks = 32
    num_tasks_per_node = 8
    num_gpus_per_node = 8
    env_vars = {
        'NCCL_DEBUG': 'INFO',
        'NCCL_SOCKET_IFNAME': 'hsn0,hsn1,hsn2,hsn3',
        'NCCL_NET_GDR_LEVEL': '3',
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

    @sanity_function
    def assert_job_is_complete(self):
        return sn.all([
            sn.assert_found(r'Launch mode Parallel/CGMD', self.stdout),
            sn.assert_found(r'Total average', self.stdout),
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
class pytorch_distr_cnn(pytorch_distr_cnn_base):
    descr = 'Check the training throughput of a cnn'
    modules = ['PyTorch']
    executable = 'python cnn_distr.py'

    tags = {'python', 'contrib/22.08'}

    # @sanity_function
    # def assert_job_is_complete(self):
    #     return sn.all([
    #         sn.assert_found(r'Using network AWS Libfabric', self.stdout),
    #         sn.assert_found(r'Selected Provider is cxi', self.stdout),
    #         super().assert_job_is_complete()
    #     ])


@rfm.simple_test
class pytorch_distr_cnn_singularity(pytorch_distr_cnn_base):
    # The container used here doesn't include all the packages needed to run
    # this test:
    # MPICC=mpicc pip install --user mpi4py
    # pip install datasets transformers python-hostlist
    descr = 'Check the training throughput of a cnn with torch.distributed'

    tags = {'singularity', 'python'}

    @run_before('run')
    def set_container_variables(self):
        self.container_platform = 'Singularity'
        self.container_platform.image = os.path.join(
            self.current_system.resourcesdir,
            'deepspeed',
            'deepspeed_rocm5.2.3_ubuntu20.04_py3.7_pytorch_1.12.1_deepspeed.sif'  # noqa: E501
        )
        self.container_platform.command = 'python cnn_distr.py'


@rfm.simple_test
class pytorch_distr_cnn_singularity_aws(pytorch_distr_cnn_singularity):
    modules = ['singularity-bindings', 'aws-ofi-rccl']

    tags = {'singularity'}

    @run_before('run')
    def set_container_variables(self):
        super().set_container_variables()
        self.container_platform.mount_points = [
            ('/appl', '/appl'),
            ('$SCRATCH', '$SCRATCH'),
            ('$EBROOTRCCL/lib/librccl.so.1.0',
             '/opt/rocm-5.0.1/rccl/lib/librccl.so.1.0.50001')
        ]
        self.env_vars.update({
            'SINGULARITYENV_LD_LIBRARY_PATH': (
                '/opt/ompi/lib:'
                '${EBROOTAWSMINOFIMINRCCL}/lib:'
                '/opt/cray/xpmem/2.4.4-2.3_9.1__gff0e1d9.shasta/lib64:'
                '${SINGULARITYENV_LD_LIBRARY_PATH}'
            ),
        })

    @sanity_function
    def assert_job_is_complete(self):
        return sn.all([
            sn.assert_found(r'Using network AWS Libfabric', self.stdout),
            sn.assert_found(r'Selected Provider is cxi', self.stdout),
            super().assert_job_is_complete()
        ])
