import os
import reframe as rfm
import reframe.utility.sanity as sn

@rfm.simple_test
class base_container_mpi(rfm.RunOnlyRegressionTest):
    descr = 'Test the singularity bindings with base container'
    valid_systems = ['lumi:small']
    valid_prog_environs = ['builtin']
    container_platform = 'Singularity'
    num_tasks = 2
    num_tasks_per_node = 1
    tags = {'container'}

    reference = {
        'lumi:small': {'bandwidth': (22180.76, -0.05, None, 'MB/s')}
    }

    @run_before('run')
    def set_container_variables(self):
        self.container_platform.image = '/appl/local/containers/sif-images/lumi-mpi4py-rocm-5.4.5-python-3.10-mpi4py-3.1.4.sif'
        self.container_platform.command = '/opt/osu/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw'
        self.env_vars = {
            'SINGULARITY_BIND':'/var/spool/slurmd:/var/spool/slurmd,'
                                '/opt/cray:/opt/cray,'
                                '/usr/lib64/libcxi.so.1:/usr/lib64/libcxi.so.1,'
                                '/usr/lib64/libjansson.so.4:/usr/lib64/libjansson.so.4'
        }

    @sanity_function
    def assert_found_max_bandwidth(self):
        max_bandwidth = r'4194304'
        return sn.assert_found(max_bandwidth, self.stdout)

    @performance_function('MB/s')
    def bandwidth(self):
        return sn.extractsingle(r'4194304\s+(?P<bw>\S+)',
                                self.stdout, 'bw', float)
