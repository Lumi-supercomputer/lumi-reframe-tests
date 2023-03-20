# https://github.com/c3sr/comm_scope
import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class comm_scope(rfm.RegressionTest):
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin-hip']
    sourcesdir = 'https://github.com/c3sr/comm_scope'
    modules = ['buildtools']
    build_system = 'CMake'
    executable = './build/comm_scope'
    maintainers = ['mszpindler']
    num_gpus_per_node = 8
    num_cpus_per_task = 8

    @run_before('compile')
    def do_cmake(self):
        self.prebuild_cmds = ['git submodule update --init --recursive']
        self.build_system.config_opts = ['-DSCOPE_ARCH_MI250X=ON', '-DSCOPE_USE_NUMA=ON']
        self.build_system.builddir = 'build'
        self.build_system.max_concurrency = 64
