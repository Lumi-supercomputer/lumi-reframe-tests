import os
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher


class tensorflow_hvd_cnn_base(rfm.RunOnlyRegressionTest):
    descr = 'Check the training throughput of a cnn with tensorflow+horovod'
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin']
    sourcesdir = 'src'
    num_tasks = 32
    num_gpus_per_node = 8
    num_tasks_per_node = num_gpus_per_node
    throughput_per_gpu = 434.2
    variables = {
        'NCCL_DEBUG': 'INFO',
        'NCCL_SOCKET_IFNAME': 'hsn0,hsn1,hsn2,hsn3',
    }

    @run_after('init')
    def set_references(self):
        throughput_total = self.throughput_per_gpu * self.num_tasks
        self.reference = {
            'lumi:gpu': {
                'samples_per_sec_per_gpu': (self.throughput_per_gpu,
                                            -0.1, None, 'samples/sec'),
                'samples_per_sec_total': (throughput_total,
                                          -0.1, None, 'samples/sec')
            }
        }

    @sanity_function
    def assert_job_is_complete(self):
        return sn.all([
            sn.assert_found(r'NCCL INFO Launch mode Parallel/CGMD',
                            self.stdout),
            sn.assert_found(r'img/sec on \d+ GPU\(s\): \S+ \+',
                            self.stdout)
        ])

    @performance_function('samples/sec')
    def samples_per_sec_per_gpu(self):
        return sn.extractsingle(
            r'Img/sec per GPU: (?P<samples_per_sec_per_gpu>\S+) \+',
            self.stdout, 'samples_per_sec_per_gpu', float
        )

    @performance_function('samples/sec')
    def samples_per_sec_total(self):
        return sn.extractsingle(
            r'img/sec on \d+ GPU\(s\): (?P<samples_per_sec_total>\S+) \+',
            self.stdout, 'samples_per_sec_total', float
        )


@rfm.simple_test
class tensorflow_hvd_cnn(tensorflow_hvd_cnn_base):
    modules = ['Horovod']
    executable = 'python tf2_hvd_synthetic_benchmark.py --batch-size=512'


@rfm.simple_test
class tensorflow_keras_hvd_cnn(tensorflow_hvd_cnn_base):
    modules = ['Horovod']
    throughput_per_gpu = 360.0
    executable = 'python tf2_keras_hvd_synthetic_benchmark.py --batch-size=512'


@rfm.simple_test
class tensorflow_hvd_cnn_singularity(tensorflow_hvd_cnn_base):
    # The container used here doesn't include all the packages needed to run
    # this test. A requirements.txt file can be found on the src directory
    throughput_per_gpu = 530.0
    modules = ['OpenMPI']
    script = 'tf2_hvd_synthetic_benchmark.py'

    @run_before('run')
    def set_container_variables(self):
        self.container_platform = 'Singularity'
        self.container_platform.image = os.path.join(
            self.current_system.resourcesdir,
            'tensorflow',
            'tensorflow_rocm5.0-tf2.7-dev.sif'
        )
        self.container_platform.command = (
            "bash -c '"
            "cd /rfm_workdir; "
            ". ~/tf2.7_rocm5.0_env/bin/activate; "
            f"python {self.script} --batch-size=512'"
        )

    @run_before('run')
    def set_launcher(self):
        # The job launcher has to be changed to `mpirun` since the software
        # in the container is based on OpenMPI and it would fail with `srun`
        self.job.launcher = getlauncher('mpirun')()


@rfm.simple_test
class tensorflow_keras_hvd_cnn_singularity(tensorflow_hvd_cnn_singularity):
    throughput_per_gpu = 481.0
    script = 'tf2_keras_hvd_synthetic_benchmark.py'


@rfm.simple_test
class tensorflow_hvd_cnn_singularity_aws(tensorflow_hvd_cnn_singularity):
    modules = ['singularity-bindings', 'rccl', 'aws-ofi-rccl', 'OpenMPI']

    @run_before('run')
    def set_container_variables(self):
        super().set_container_variables()
        self.container_platform.mount_points = [
            ('/appl', '/appl'),
            ('$EBROOTRCCL/lib/librccl.so.1.0',
             '/opt/rocm-5.0.0/rccl/lib/librccl.so.1.0.50000')
        ]
        self.variables.update({
            'SINGULARITYENV_LD_LIBRARY_PATH': (
                '/openmpi/lib:/opt/rocm-5.0.0/lib:'
                '$EBROOTAWSMINOFIMINRCCL/lib:'
                '/opt/cray/xpmem/2.4.4-2.3_9.1__gff0e1d9.shasta/lib64:'
                '$SINGULARITYENV_LD_LIBRARY_PATH'
            ),
        })

    @sanity_function
    def assert_job_is_complete(self):
        return sn.all([
            sn.assert_found(r'Using network AWS Libfabric', self.stdout),
            sn.assert_found(r'Selected Provider is cxi', self.stdout),
            super().assert_job_is_complete()
        ])


@rfm.simple_test
class tensorflow_keras_hvd_cnn_singularity_aws(tensorflow_hvd_cnn_singularity_aws):  # noqa: E501
    throughput_per_gpu = 481.0
    script = 'tf2_keras_hvd_synthetic_benchmark.py'
