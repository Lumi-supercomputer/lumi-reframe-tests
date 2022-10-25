import os
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.utility.osext import cray_cdt_version


class singularity_cray_mpich_bindings_test(rfm.RunOnlyRegressionTest):
    descr = 'Test the singularity-bindings module with glibc'
    valid_systems = ['lumi:gpu', 'lumi:standard']
    valid_prog_environs = ['builtin']
    container_platform = 'Singularity'
    num_tasks = 2
    num_tasks_per_node = 1
    reference = {
        'lumi:gpu': {'bandwidth': (22180.76, -0.05, None, 'MB/s')}
    }

    @run_before('run')
    def prepare_build(self):
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'singularity-binding-test')

    @sanity_function
    def assert_found_max_bandwidth(self):
        max_bandwidth = r'4194304'
        return sn.assert_found(max_bandwidth, self.stdout)

    @performance_function('MB/s')
    def bandwidth(self):
        return sn.extractsingle(r'4194304\s+(?P<bw>\S+)',
                                self.stdout, 'bw', float)


@rfm.simple_test
class test_glibc_mount(singularity_cray_mpich_bindings_test):
    modules = ['singularity-bindings']

    @run_before('run')
    def set_container_variables(self):
        self.container_platform.image = 'osu-debian-jessie.sif'
        self.container_platform.command = '/home/osu/p2p_osu_bw'


@rfm.simple_test
class test_no_glibc_mount(singularity_cray_mpich_bindings_test):
    pe_version = cray_cdt_version()
    modules = [f'singularity-bindings/system-cpeGNU-{pe_version}-noglibc']

    @run_before('run')
    def set_container_variables(self):
        self.container_platform.image = 'osu-ubuntu-21.04.sif'
        self.container_platform.command = (
            '/usr/local/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw'
        )
