import os
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher


class fetch_rccl(rfm.RunOnlyRegressionTest):
    '''Fixture for fetching RCCL.'''
    local = True
    executable = 'git clone git@github.com:ROCmSoftwarePlatform/rccl.git'  # noqa: E501

    @sanity_function
    def validate_download(self):
        return sn.assert_eq(self.job.exitcode, 0)


class fetch_rccl_tests(rfm.RunOnlyRegressionTest):
    '''Fixture for fetching the rccl-tests.'''
    local = True
    executable = 'git clone git@github.com:ROCmSoftwarePlatform/rccl-tests.git'  # noqa: E501

    @sanity_function
    def validate_download(self):
        return sn.assert_eq(self.job.exitcode, 0)


class fetch_aws_ofi_rccl_plugin(rfm.RunOnlyRegressionTest):
    '''Fixture for fetching the AWS libfabric plugin.'''
    local = True
    executable = 'git clone git@github.com:ROCmSoftwarePlatform/aws-ofi-rccl.git'  # noqa: E501

    @sanity_function
    def validate_download(self):
        return sn.assert_eq(self.job.exitcode, 0)


class rccl_test_base(rfm.RunOnlyRegressionTest):
    rccl_dir = '/rccl_stage'
    rccl_tests_dir = '/rccl_tests_stage'
    aws_plugin_dir = '/aws_plugin_stage'

    def set_container_platform(self):
        self.container_platform = 'Singularity'
        self.container_platform.image = os.path.join(
            self.current_system.resourcesdir,
            'deepspeed',
            'deepspeed_rocm5.0.1_ubuntu18.04_py3.7_pytorch_1.10.0.sif'
        )


class build_rccl(rccl_test_base):
    '''Fixture for building the OSU benchmarks'''
    local = True
    rccl = fixture(fetch_rccl, scope='session')

    @run_before('run')
    def set_container_variables(self):
        super().set_container_platform()
        self.container_platform.mount_points = [
            (self.rccl.stagedir, self.rccl_dir),
        ]
        self.container_platform.command = (
            "bash -c '"
            f"cd {self.rccl_dir}/rccl; "
            "mkdir build;"
            "cd build;"
            "CXX=/opt/rocm-5.0.1/bin/hipcc cmake -DCMAKE_PREFIX_PATH=./install ..;"  # noqa: E501
            "make -j 16"
            "'"
        )

    @sanity_function
    def validate_build(self):
        return sn.assert_found(r'\[100%\] Built target rccl', self.stdout)


class build_rccl_tests(rccl_test_base):
    '''Fixture for building the OSU benchmarks'''
    local = True
    rccl = fixture(fetch_rccl, scope='session')
    rccl_tests = fixture(fetch_rccl_tests, scope='session')
    rccl_binaries = fixture(build_rccl, scope='session')

    @sanity_function
    def validate_build(self):
        num_binaries = sn.count(
            sn.glob(f'{self.rccl_tests.stagedir}/rccl-tests/build/*')
        )
        return sn.assert_eq(num_binaries, 11)

    @run_before('run')
    def set_container_variables(self):
        super().set_container_platform()
        self.container_platform.mount_points = [
            (self.rccl.stagedir, self.rccl_dir),
            (self.rccl_tests.stagedir, self.rccl_tests_dir),
        ]
        self.container_platform.command = (
            "bash -c '"
            f"cd {self.rccl_tests_dir}/rccl-tests; "
            f"make MPI=1 MPI_HOME=/opt/ompi HIP_HOME=/opt/rocm-5.0.1/hip RCCL_HOME={self.rccl_dir}/rccl/build -j 16"  # noqa: E501
            "'"
        )


class build_aws_plugin(rccl_test_base):
    '''Fixture for building the OSU benchmarks'''
    rccl = fixture(fetch_rccl, scope='session')
    rccl_tests = fixture(fetch_rccl_tests, scope='session')
    aws_plugin = fixture(fetch_aws_ofi_rccl_plugin, scope='session')
    rccl_binaries = fixture(build_rccl, scope='session')
    modules = ['singularity-bindings/system-cpeGNU-22.06-noglibc']

    @sanity_function
    def validate_build(self):
        return sn.path_isfile(f'{self.aws_plugin.stagedir}/aws-ofi-rccl/'
                              'install/lib/librccl-net.so.0.0.0')

    @run_before('run')
    def set_container_variables(self):
        super().set_container_platform()
        self.container_platform.mount_points = [
            (self.rccl.stagedir, self.rccl_dir),
            (self.rccl_tests.stagedir, self.rccl_tests_dir),
            (self.aws_plugin.stagedir, self.aws_plugin_dir),
        ]
        self.container_platform.command = (
            "bash -c '"
            f"cd {self.aws_plugin_dir}/aws-ofi-rccl; "
            "./autogen.sh;"
            f"CC=mpicc ./configure --with-libfabric=/opt/cray/libfabric/1.15.0.0 "  # noqa: E501
            f"                     --with-hip=/opt/rocm-5.0.1 "
            f"                     --with-rccl={self.rccl_dir}/rccl/build "
            f"                     --with-mpi=/opt/ompi "
            f"                     --prefix=$PWD/install;"
            "make;"
            "make install"
            "'"
        )


@rfm.simple_test
class rccl_tests_allreduce(rccl_test_base):
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin']
    modules = ['singularity-bindings/system-cpeGNU-22.06']
    num_tasks = 16
    num_tasks_per_node = 8
    num_gpus_per_node = 8
    rccl = fixture(fetch_rccl, scope='session')
    rccl_tests = fixture(fetch_rccl_tests, scope='session')
    aws_plugin = fixture(fetch_aws_ofi_rccl_plugin, scope='session')
    rccl_binaries = fixture(build_rccl, scope='session')
    rccl_tests_binaries = fixture(build_rccl_tests, scope='session')
    aws_plugin_binaries = fixture(build_aws_plugin, scope='session')
    reference = {
        'lumi:gpu': {
            'busbw': (84.76, -0.05, None, 'GB/s'),
            'algbw': (45.21, -0.05, None, 'GB/s'),
        }
    }

    @run_after('init')
    def set_variables(self):
        self.variables = {
            'NCCL_DEBUG': 'INFO',
            'NCCL_NET_GDR_LEVEL': '3',
            'SINGULARITYENV_LD_LIBRARY_PATH': f'/opt/rocm-5.0.1/lib:'
                                              f'{self.rccl_dir}/rccl/build/:'
                                              f'{self.aws_plugin_dir}/aws-ofi-rccl/install/lib:'  # noqa: E501
                                              '/opt/cray/xpmem/2.3.2-2.2_6.13__g93dd7ee.shasta/lib64:'  # noqa: E501
                                              '/opt/ompi/lib:'
                                              '$SINGULARITYENV_LD_LIBRARY_PATH'
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

    @run_before('run')
    def set_container_platform(self):
        super().set_container_platform()
        self.container_platform.mount_points = [
            (self.rccl.stagedir, self.rccl_dir),
            (self.rccl_tests.stagedir, self.rccl_tests_dir),
            (self.aws_plugin.stagedir, self.aws_plugin_dir),
        ]
        self.container_platform.command = (
            "bash -c '"
            "export LD_LIBRARY_PATH=/rccl_stage/rccl/build:${LD_LIBRARY_PATH};"
            f"{self.rccl_tests_dir}/rccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1"  # noqa: E501
            "'"
        )

    @run_before('run')
    def set_launcher(self):
        # The job launcher has to be changed to `mpirun` since the software
        # in the container is based on OpenMPI and it would fail with `srun`
        self.job.launcher = getlauncher('mpirun')()
