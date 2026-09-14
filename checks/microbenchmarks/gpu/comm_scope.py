# This check uses code from https://github.com/c3sr/comm_scope
import os
import reframe as rfm
import reframe.utility.sanity as sn

class lumi_build_comm_scope(rfm.CompileOnlyRegressionTest):
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin']
    sourcesdir = 'https://github.com/mszpindler/comm_scope -b updates_for_lumi'
    modules = ['buildtools', 'rocm']
    build_system = 'CMake'

    @run_before('compile')
    def do_cmake(self):
        self.prebuild_cmds = ['git submodule update --init --recursive']
        self.build_system.config_opts = ['--fresh', '-DCMAKE_C_COMPILER=amdclang', '-DCMAKE_CXX_COMPILER=hipcc', '-DSCOPE_ARCH_MI250X=ON', '-DSCOPE_USE_NUMA=ON', '-DCMAKE_CXX_FLAGS="-D__HIP_PLATFORM_AMD__ -fopenmp"']
        self.build_system.builddir = 'build'
        self.build_system.max_concurrency = 16

    @sanity_function
    def check_build(self):
        return sn.assert_found('Built target comm_scope', self.stdout)

class lumi_comm_scope(rfm.RunOnlyRegressionTest):
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin']
    modules = ['rocm']
    maintainers = ['mszpindler']
    num_gpus_per_node = 8
    num_cpus_per_task = 8

    topology = parameter(['quad', 'dual', 'single'], loggable=True) 
    perf_relative = variable(float, value=0.0, loggable=True)

    build_comm_scope = fixture(lumi_build_comm_scope, scope='environment')

    tags = {'production', 'craype'}

    @run_before('run')
    def set_gpu_affinity(self):
        self.src_gpu = 0
        match self.topology:
            case 'quad':
                self.dst_gpu = 1
            case 'dual':
                self.dst_gpu = 6
            case 'single':
                self.dst_gpu = 2

    @run_before('performance')
    def set_perf_variables(self):

        self.perf_variables = {
            self.topology: self.bytes_per_second(self.topology)
            #'quad': self.bytes_per_second(),
            #'dual': self.bytes_per_second(6),
            #'single': self.bytes_per_second(2),
        }

    @run_after('performance')
    def higher_the_better(self):
        perf_var = self.topology
        key_str = self.current_partition.fullname+':'+perf_var
        try:
            found = self.perfvalues[key_str]
        except KeyError:
            return None

        if self.perfvalues[key_str][1] != 0:
            self.perf_relative = ((self.perfvalues[key_str][0]-self.perfvalues[key_str][1])/self.perfvalues[key_str][1])

@rfm.simple_test
class comm_implicit_managed(lumi_comm_scope):
    # Reference numbers are taken from C. Pearson "Interconnect Bandwidth Heterogeneity on AMD MI250x and Infinity Fabric" (https://arxiv.org/pdf/2302.14827.pdf)
    reference = {
        'lumi:gpu': {
            'quad':   (148, -0.05, 0.05, 'GB/s'),
            'dual':   (76, -0.05, 0.05, 'GB/s'),
            'single': (38, -0.05, 0.05, 'GB/s'),
        }
    }

    @run_before('run')
    def set_executable_opts(self):
        self.executable = os.path.join(self.build_comm_scope.stagedir, 'build', 'comm_scope')
        self.executable_opts = [f'--benchmark_filter="Comm_implicit_managed_GPUWrGPU_fine/{self.src_gpu}/{self.dst_gpu}/log2\(N\):30/"', '--benchmark_out_format=json', '--benchmark_out=rfm_job.json']

    @run_before('run')
    def set_env_vars(self):
        self.env_vars = {
            'HSA_ENABLE_SDMA': 1,
            #'HSA_XNACK': 1,
        }

    @sanity_function
    def validate_benchmarks(self):
        return sn.assert_found('dst_gpu', 'rfm_job.json')

    @performance_function('GB/s')
    def bytes_per_second(self, connection='quad'):
        bps = sn.extractsingle(rf'Comm_implicit_managed_GPUWrGPU_fine\S+.*bytes_per_second=(\S+)G\/s\s+dst_gpu={self.dst_gpu}\s+src_gpu=0', self.stdout, 1, float)
        # The benchmark formats bps on standard output in GiB not GB
        return bps*(1024**3)*1e-9


@rfm.simple_test
class comm_hipmemcpy_peer(lumi_comm_scope):
    reference = {
        'lumi:gpu': {
            'quad':   (177, -0.05, 0.05, 'GB/s'),
            'dual':   (88, -0.05, 0.05, 'GB/s'), 
            'single': (44, -0.05, 0.05, 'GB/s'),
        }
    }

    tags = {'production', 'craype'}

    @run_before('run')
    def set_executable_opts(self):
        self.executable = os.path.join(self.build_comm_scope.stagedir, 'build', 'comm_scope')
        self.executable_opts = [f'--benchmark_filter="Comm_hipMemcpyPeerAsync_GPUToGPUPeer/{self.src_gpu}/{self.dst_gpu}/log2\(N\):30/"', '--benchmark_out_format=json', '--benchmark_out=rfm_job.json']

    @run_before('run')
    def set_env_vars(self):
        self.env_vars = {
            'HSA_ENABLE_SDMA': 0,
            #'HSA_XNACK': 1,
        }

    @sanity_function
    def validate_benchmarks(self):
        return sn.assert_found('dstGpu', 'rfm_job.json')

    @performance_function('GB/s')
    def bytes_per_second(self, connection='quad'):
        bps = sn.extractsingle(rf'Comm_hipMemcpyPeerAsync_GPUToGPUPeer\S+.*bytes_per_second=(\S+)G\/s\s+dstGpu={self.dst_gpu}\s+srcGpu=0', self.stdout, 1, float)
        # The benchmark formats bps on standard output in GiB not GB
        return bps*(1024**3)*1e-9
