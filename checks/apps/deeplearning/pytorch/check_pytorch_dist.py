import os
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher


@rfm.simple_test
class pytorch_distr_cnn(rfm.RunOnlyRegressionTest):
    # The container used here doesn't include all the packages needed to run
    # this test:
    # MPICC=mpicc pip install --user mpi4py
    # pip install datasets transformers python-hostlist
    descr = 'Check the training throughput of a cnn with torch.distributed'
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin']
    modules = ['singularity-bindings']
    sourcesdir = 'src'
    num_tasks = 4
    num_tasks_per_node = 1
    num_gpus_per_node = 8
    variables = {
        'NCCL_DEBUG': 'INFO'
    }
    throughput_per_gpu = 567.65
    throughput_total = throughput_per_gpu * num_tasks * num_gpus_per_node
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
            'deepspeed',
            'deepspeed_rocm5.0.1_ubuntu18.04_py3.7_pytorch_1.10.0.sif'
        )
        self.container_platform.command = (
            "bash -c '"
            "cd /rfm_workdir; "
            "python cnn_distr.py'"
        )

    @sanity_function
    def assert_found_nccl_launch(self):
        return sn.assert_found(r'NCCL INFO Launch mode Parallel/CGMD',
                               self.stdout)

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