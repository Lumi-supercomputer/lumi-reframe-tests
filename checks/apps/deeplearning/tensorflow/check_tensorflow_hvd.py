import os
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher


@rfm.simple_test
class tensorflow_hvd_cnn(rfm.RunOnlyRegressionTest):
    # The container used here doesn't include all the packages needed to run
    # this test. A requirements.txt file can be found on the src directory
    descr = 'Check the training throughput of a cnn with tensorflow+horovod'
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin']
    modules = ['singularity-bindings']
    sourcesdir = 'src'
    num_tasks = 32
    num_gpus_per_node = 8
    num_tasks_per_node = num_gpus_per_node
    variables = {
        'NCCL_DEBUG': 'INFO',
    }
    throughput_per_gpu = 506.55
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
    def set_container_variables(self):
        self.container_platform = 'Singularity'
        self.container_platform.image = os.path.join(
            self.current_system.resourcesdir,
            'tensorflow',
            'tensorflow_rocm5.2.0-tf2.9-dev.sif'
        )
        self.container_platform.command = (
            "bash -c '"
            "cd /rfm_workdir; "
            "python tf2_hvd_synthetic_benchmark.py --batch-size=256'"
        )

    @run_before('run')
    def set_launcher(self):
        # The job launcher has to be changed to `mpirun` since the software
        # in the container is based on OpenMPI and it would fail with `srun`
        self.job.launcher = getlauncher('mpirun')()

    @sanity_function
    def assert_found_nccl_launch(self):
        return sn.assert_found(r'NCCL INFO Launch mode Parallel/CGMD',
                               self.stdout)

    @performance_function('samples/sec')
    def samples_per_sec_per_gpu(self):
        return sn.avg(sn.extractall(
            r'Iter #\d+: (?P<samples_per_sec_per_gpu>\S+) img',
            self.stdout, 'samples_per_sec_per_gpu', float
        ))

    @performance_function('samples/sec')
    def samples_per_sec_total(self):
        return sn.avg(sn.extractall(
            r'img/sec on \d+ GPU\(s\): (?P<samples_per_sec_total>\S+) \+',
            self.stdout, 'samples_per_sec_total', float
        ))
