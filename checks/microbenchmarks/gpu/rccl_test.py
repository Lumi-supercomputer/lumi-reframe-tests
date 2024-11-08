import os
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher
from reframe.utility.osext import cray_cdt_version

@rfm.simple_test
class rccl_test_allreduce(rfm.RegressionTest):
    descr = 'Compile and run rccl-test'
    build_system = 'CMake'
    repo_name = 'rccl-tests'
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['cpeGNU']
    modules =['rocm', 'buildtools', 'aws-ofi-rccl']
    num_tasks = 16
    num_tasks_per_node = 8
    num_gpus_per_node = 8
    executable_opts = ['-b 2M', '-e 2048M', '-f 2', '-g 1', '-z 1', '-t 1', '-R 1', '-n 80', '-w 5', '-d half']
    executable = 'build/all_reduce_perf'
    exclusive_access = True

    reference = {
        'lumi:gpu': {
            'busbw': (85.00, -0.05, None, 'GB/s'),
            'algbw': (45.00, -0.05, None, 'GB/s'),
        }
    }

    @run_before('compile')
    def set_compiler_flags(self):
        self.sourcesdir = f'https://github.com/ROCmSoftwarePlatform/{self.repo_name}'
        self.build_system.builddir = 'build'
        self.build_system.config_opts = ['--fresh', '-DMPI_MPICXX=CC', '-DCMAKE_CXX_COMPILER=hipcc', '-DCMAKE_CXX_FLAGS="--offload-arch=gfx90a"','-DGPU_TARGETS=gfx90a', '-DMPI_PATH=$CRAY_MPICH_DIR', '-DCMAKE_EXE_LINKER_FLAGS="$PE_MPICH_GTL_DIR_amd_gfx90a -lmpi_gtl_hsa"']
        self.build_system.make_opts = ['VERBOSE=1', '-j8']

    @run_after('init')
    def set_variables(self):
        self.env_vars = {
            'NCCL_DEBUG': 'INFO',
            'NCCL_SOCKET_IFNAME': 'hsn0,hsn1,hsn2,hsn3',
            'NCCL_NET_GDR_LEVEL': '3',
            'NCCL_ENABLE_DMABUF_SUPPORT': '1',
            'MPICH_GPU_SUPPORT_ENABLED': '1',
        }
    
    @run_before('run')
    def set_cpu_binding(self):
         self.job.launcher.options = ['--cpu-bind="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"']

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

@rfm.simple_test
class rccl_test_allreduce_containerized(rfm.RunOnlyRegressionTest):
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin']
    container_platform = 'Singularity'
    num_tasks = 16
    num_tasks_per_node = 8
    num_gpus_per_node = 8
    exclusive_access = True
    cont_image = parameter(['6.0.3-dockerhash-1ada3e019646', '6.2.0-dockerhash-c8e37dff2e91', '6.1.3-dockerhash-4a063f050ed7'])

    reference = {
        'lumi:gpu': {
            'busbw': (85.00, -0.05, None, 'GB/s'),
            'algbw': (45.00, -0.05, None, 'GB/s'),
        }
    }

    @run_before('run')
    def set_launch_settings(self):
        self.container_platform.image = os.path.join(
                self.current_system.resourcesdir,
                'containers',
                f'lumi-rocm-rocm-{self.cont_image}.sif'
                )
        self.container_platform.command = '/opt/rccltests/all_reduce_perf -z 1 -b 2M -e 2048M -f 2 -g 1 -t 1 -R 1 -n 80 -w 5 -d half'
        self.env_vars = {
            'NCCL_DEBUG': 'INFO',
            'NCCL_SOCKET_IFNAME': 'hsn0,hsn1,hsn2,hsn3',
            'NCCL_NET_GDR_LEVEL': '3',
            'CXI_FORK_SAFE': '1',
            'CXI_FORK_SAFE_HP': '1',
            'FI_CXI_DISABLE_CQ_HUGETLB': '1',
            'SINGULARITY_BIND':'/var/spool/slurmd:/var/spool/slurmd,'
                                '/opt/cray:/opt/cray,'
                                '/usr/lib64/libcxi.so.1:/usr/lib64/libcxi.so.1,'
                                '/usr/lib64/libjansson.so.4:/usr/lib64/libjansson.so.4'
        }

    @run_before('run')
    def set_cpu_binding(self):
         self.job.launcher.options = ['--cpu-bind="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"']

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
