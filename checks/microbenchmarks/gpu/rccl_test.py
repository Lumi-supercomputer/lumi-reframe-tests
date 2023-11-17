import os
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher
from reframe.utility.osext import cray_cdt_version

@rfm.simple_test
class rccl_test(rfm.RegressionTest):
    #COMPILATION
    descr = 'Compile and run rccl-test'
    build_system = 'CMake'
    repo_name = 'rccl-tests'

    #RUN
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin']
    modules = ['LUMI/22.12','partition/G', 'cpeGNU', 'rocm', 'cray-mpich', 'buildtools/22.12', 'aws-ofi-rccl']
    num_tasks = 16
    num_tasks_per_node = 8
    num_gpus_per_node = 8
    executable_opts = ['-b 8', '-e 128M', '-f 2', '-g 1']
    executable = 'build/all_reduce_perf'
    #prebuild_cmds = ['module list', 'env']

    reference = {
        'lumi:gpu': {
            'busbw': (70.00, -0.05, None, 'GB/s'),
            'algbw': (40.00, -0.05, None, 'GB/s'),
        }
    }

    @run_before('compile')
    def set_compiler_flags(self):
        self.sourcesdir = f'https://github.com/ROCmSoftwarePlatform/{self.repo_name}'
        self.build_system.builddir = 'build'
        self.build_system.config_opts = ['--fresh', '-DMPI_MPICXX=CC', '-DCMAKE_CXX_COMPILER=hipcc', '-DCMAKE_CXX_FLAGS="--offload-arch=gfx90a --amdgpu-target=gfx90a"','-DGPU_TARGETS=gfx90a', '-DMPI_PATH=$CRAY_MPICH_DIR']
        #self.build_system.flags_from_environ = False
        self.build_system.make_opts = ['VERBOSE=1', '-j8']

    @run_after('init')
    def set_variables(self):
        self.env_vars = {
            'NCCL_DEBUG': 'INFO',
            'NCCL_SOCKET_IFNAME': 'hsn0,hsn1,hsn2,hsn3',
            'NCCL_NET_GDR_LEVEL': '3',
        }

    @sanity_function
    def check_last_line(self):
        return sn.assert_found(r'Avg bus bandwidth', self.stdout)

    @performance_function('GB/s')
    def busbw(self):
        return sn.extractsingle(
            r'^\s+134217728.+\s+(?P<busbw>\S+)\s+\S+$',
            self.stdout, 'busbw', float
        )

    @performance_function('GB/s')
    def algbw(self):
        return sn.extractsingle(
            r'^\s+134217728.+\s+(?P<algbw>\S+)\s+\S+\s+\S+$',
            self.stdout, 'algbw', float
        )
