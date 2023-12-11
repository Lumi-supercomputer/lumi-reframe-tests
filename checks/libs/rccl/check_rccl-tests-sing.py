import os
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher
from reframe.utility.osext import cray_cdt_version


class fetch_rccl(rfm.RunOnlyRegressionTest):
    '''Fixture for fetching RCCL.'''
    local = True
    url_tag = 'rocm-5.2.3.tar.gz'
    file_name = f'rccl-{url_tag}'
    executable = f'curl -LJO https://github.com/ROCmSoftwarePlatform/rccl/archive/refs/tags/{url_tag}'  # noqa: E501
    postrun_cmds = [
        f'tar xzf {file_name}'
    ]

    @sanity_function
    def validate_download(self):
        return sn.assert_eq(self.job.exitcode, 0)


class fetch_rccl_tests(rfm.RunOnlyRegressionTest):
    '''Fixture for fetching the rccl-tests.'''
    local = True
    repo_name = 'rccl-tests'
    executable = f'git clone -b develop http://github.com/ROCmSoftwarePlatform/{repo_name}.git'  # noqa: E501
    postrun_cmds = [
        f'cd {repo_name};'
        'git checkout 3fbd328'
    ]

    @sanity_function
    def validate_download(self):
        return sn.assert_eq(self.job.exitcode, 0)


class fetch_aws_ofi_rccl_plugin(rfm.RunOnlyRegressionTest):
    '''Fixture for fetching the AWS libfabric plugin.'''
    local = True
    repo_name = 'aws-ofi-rccl'
    executable = f'git clone -b cxi http://github.com/ROCmSoftwarePlatform/{repo_name}.git'  # noqa: E501
    postrun_cmds = [
        f'cd {repo_name};'
        'git checkout 66b3b31'
    ]

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
            'deepspeed_rocm5.2.3_ubuntu20.04_py3.7_pytorch_1.12.1_deepspeed.sif'  # noqa: E501
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
            f"cd {self.rccl_dir}/{self.rccl.file_name[:-7]}; "
            "mkdir build;"
            "cd build;"
            "CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_PREFIX_PATH=./install ..;"  # noqa: E501
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
        build_dir = os.path.join(
            self.rccl_tests.stagedir,
            self.rccl_tests.repo_name,
            'build'
        )
        num_binaries = sn.count(
            sn.glob(f'{build_dir}/*')
        )
        return sn.assert_eq(num_binaries, 10)

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
            f"make MPI=1 MPI_HOME=/opt/ompi HIP_HOME=/opt/rocm/hip RCCL_HOME={self.rccl_dir}/rccl/build -j 16"  # noqa: E501
            "'"
        )


class build_aws_plugin(rccl_test_base):
    '''Fixture for building the OSU benchmarks.

    This needs to be built on a compute node since it needs
    network libraries only available there.
    '''
    rccl = fixture(fetch_rccl, scope='session')
    rccl_tests = fixture(fetch_rccl_tests, scope='session')
    aws_plugin = fixture(fetch_aws_ofi_rccl_plugin, scope='session')
    rccl_binaries = fixture(build_rccl, scope='session')
    pe_version = cray_cdt_version()
    modules = [f'singularity-bindings/system-cpeGNU-{pe_version}-noglibc']

    @sanity_function
    def validate_build(self):
        lib_dir = os.path.join(
            self.aws_plugin.stagedir,
            self.aws_plugin.repo_name,
            'install', 'lib'
        )
        return sn.path_isfile(f'{lib_dir}/librccl-net.so.0.0.0')

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
            f"                     --with-hip=/opt/rocm "
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
    modules = ['OpenMPI', 'singularity-bindings']
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

    tags = {'singularity', 'contrib/22.06', 'contrib/22.08'}

    @run_after('init')
    def set_variables(self):
        self.env_vars = {
            'NCCL_DEBUG': 'INFO',
            'NCCL_SOCKET_IFNAME': 'hsn0,hsn1,hsn2,hsn3',
            'NCCL_NET_GDR_LEVEL': '3',
            'SINGULARITYENV_LD_LIBRARY_PATH': f'/opt/rocm/lib:'
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
