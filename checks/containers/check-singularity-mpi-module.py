import os
import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class SingularityCrayMPICHBindingsTest(rfm.RunOnlyRegressionTest):
    descr = 'Test the singularity-bindings module with glibc'
    valid_systems = ['lumi:gpu', 'lumi:standard']
    valid_prog_environs = ['builtin']
    num_tasks = 2
    num_tasks_per_node = 1
    modules = ['singularity-bindings']
    reference = {
        'lumi:gpu': {'bandwidth': (22180.76, -0.05, None, 'MB/s')}
    }

    @run_before('run')
    def prepare_build(self):
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'singularity-binding-test')

    @run_before('run')
    def set_container_variables(self):
        self.container_platform = 'Singularity'
        self.container_platform.image = 'osu-debian-jessie.sif'
        self.container_platform.command = '/home/osu/p2p_osu_bw'

    @sanity_function
    def assert_release(self):
        max_bandwidth = r'4194304'
        return sn.assert_found(max_bandwidth, self.stdout)

    @performance_function('MB/s')
    def bandwidth(self):
        return sn.extractsingle(r'4194304\s+(?P<bw>\S+)',
                                self.stdout, 'bw', float)
