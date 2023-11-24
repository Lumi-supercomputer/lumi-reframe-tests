# This check uses code from https://github.com/c3sr/comm_scope
import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class comm_scope(rfm.RegressionTest):
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin']
    sourcesdir = 'https://github.com/c3sr/comm_scope -b v0.12.0'
    modules = ['buildtools', 'rocm']
    build_system = 'CMake'
    executable = './build/comm_scope'
    executable_opts = ['--benchmark_filter="Comm_implicit_managed_GPUWrGPU_fine/0/([1-7])/log2\(N\):30/"', '--benchmark_out_format=json', '--benchmark_out=rfm_job.json']
    maintainers = ['mszpindler']
    num_gpus_per_node = 8
    num_cpus_per_task = 8

    # Reference numbers are taken from C. Pearson "Interconnect Bandwidth Heterogeneity on AMD MI250x and Infinity Fabric" (https://arxiv.org/pdf/2302.14827.pdf)
    reference = {
        'lumi:gpu': {
            'quad':   (148, -0.05, 0.05, 'GB/s'),
            'dual':   (76, -0.05, 0.05, 'GB/s'),
            'single': (38, -0.05, 0.05, 'GB/s'),
        }
    }

    tags = {'production', 'craype'}

    @run_after('setup')
    def setup_compile(self):
        self.build_job.num_cpus_per_task = 64

    @run_before('run')
    def set_env_vars(self):
        self.env_vars = {
            'LD_LIBRARY_PATH': '$LD_LIBRARY_PATH:/opt/rocm/llvm/lib/',
        }

    @run_before('compile')
    def do_cmake(self):
        self.prebuild_cmds = ['git submodule update --init --recursive']
        self.build_system.config_opts = ['--fresh', '-DCMAKE_CXX_COMPILER=hipcc', '-DSCOPE_ARCH_MI250X=ON', '-DSCOPE_USE_NUMA=ON', '-DCMAKE_CXX_FLAGS="-D__HIP_PLATFORM_AMD__"']
        self.build_system.flags_from_environ = False
        self.build_system.builddir = 'build'
        self.build_system.max_concurrency = 8

    @run_before('performance')
    def set_perf_variables(self):

        self.perf_variables = {
            'quad': self.bytes_per_second(),
            'dual': self.bytes_per_second(6),
            'single': self.bytes_per_second(2),
        }

    @sanity_function
    def validate_benchmarks(self):
        return sn.assert_eq(sn.count( sn.findall('dst_gpu', 'rfm_job.json') ), self.num_gpus_per_node-1)

    @performance_function('GB/s')
    def bytes_per_second(self, dst_gpu=1):
        bps = sn.extractsingle(rf'Comm_implicit_managed_GPUWrGPU_fine\S+.*bytes_per_second=(\S+)G\/s\s+dst_gpu={dst_gpu}\s+src_gpu=0', self.stdout, 1, float)
        # The benchmark formats bps on standard output in GiB not GB
        return bps*(1024**3)*1e-9

