# https://github.com/c3sr/comm_scope
import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class comm_scope(rfm.RegressionTest):
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin-hip']
    sourcesdir = 'https://github.com/c3sr/comm_scope -b v0.12.0'
    modules = ['buildtools']
    build_system = 'CMake'
    executable = './build/comm_scope'
    executable_opts = ['--benchmark_filter="Comm_implicit_managed_GPUWrGPU_fine/0/([1-7])/log2\(N\):30/"', '--benchmark_out_format=json', '--benchmark_out=rfm_job.json']
    maintainers = ['mszpindler']
    num_gpus_per_node = 8
    num_cpus_per_task = 8

    # Reference numbers are taken from C. Pearson "Interconnect Bandwidth Heterogeneity on AMD MI250x and Infinity Fabric" (https://arxiv.org/pdf/2302.14827.pdf)
    reference = {
        'lumi:gpu': {
            'quad':   (148, -0.1, 0.1, 'GB/s'),
            'dual':   (76, -0.1, 0.1, 'GB/s'),
            'single': (38, -0.1, 0.1, 'GB/s'),
        }
    }

    @run_after('setup')
    def setup_compile(self):
        self.build_job.num_cpus_per_task = 64

    @run_before('run')
    def set_env_vars(self):
        self.variables = {
            'LD_LIBRARY_PATH': '$LD_LIBRARY_PATH:/opt/rocm/llvm/lib/',
        }

    @run_before('compile')
    def do_cmake(self):
        self.prebuild_cmds = ['git submodule update --init --recursive']
        self.build_system.config_opts = ['--fresh', '-DCMAKE_CXX_COMPILER=hipcc', '-DSCOPE_ARCH_MI250X=ON', '-DSCOPE_USE_NUMA=ON', '-DCMAKE_CXX_FLAGS="-D__HIP_PLATFORM_AMD__"']
        self.build_system.flags_from_environ = False
        self.build_system.builddir = 'build'
        self.build_system.max_concurrency = 8

    @sanity_function
    def validate_benchmarks(self):
        return sn.assert_eq(sn.count( sn.findall('dst_gpu', 'rfm_job.json') ), self.num_gpus_per_node-1)

    @performance_function('B/s')
    def bytes_per_second(self, kind='quad'):

        if kind not in ('quad', 'dual', 'single'):
            raise ValueError(f'illegal value in argument kind ({kind!r})')

        return sn.extractsingle(r'\s+.*"bytes_per_second"\:\s+(?P<bps>\S+),', 'rfm_job.json', 'bps', float)
