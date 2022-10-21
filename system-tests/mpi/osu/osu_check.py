import contextlib
import os
import reframe as rfm
import reframe.utility.sanity as sn
from hpctestlib.microbenchmarks.mpi.osu import (build_osu_benchmarks,
                                                fetch_osu_benchmarks,
                                                osu_build_run)

class lumi_fetch_osu_benchmarks(rfm.RunOnlyRegressionTest):
   # This test implies version 6.0 or later due to code structure change
   # introduced in the version 6.0 of OSU microbenchmarks
   version = variable(str, value='6.0')
   local = True

   @run_before('run')
   def fetch(self):
       osu_file_name = f'osu-micro-benchmarks-{self.version}.tar.gz'
       self.executable = f'curl -LJO http://mvapich.cse.ohio-state.edu/download/mvapich/{osu_file_name}'

   @sanity_function
   def validate_download(self):
       return sn.assert_eq(self.job.exitcode, 0)
   

class lumi_build_osu_benchmarks(build_osu_benchmarks):
    build_type = parameter(['cpu', 'rocm'])
    osu_benchmarks = fixture(lumi_fetch_osu_benchmarks, scope='session', variables={'version': '6.1'})

    @run_after('init')
    def setup_modules(self):
        if self.build_type == 'rocm':
            self.modules = ['rocm']

    @run_before('compile')
    def prepare_make(self):
        # update directory structure
        self.build_system.make_opts = ['-C', 'c/mpi']
        self.build_system.cflags = ['-Wno-return-type', '-I$MPICH_DIR/include']

class lumi_osu_benchmarks(osu_build_run):
    tags = {'production', 'benchmark',}
    maintainers = ['@rsarm', '@mszpindler']

    @run_before('run')
    def add_exec_prefix(self):
        build_type = self.osu_binaries.build_type
        bench_path = self.benchmark_info[0].replace('.', '/')
        # update directory structure
        self.executable = os.path.join(self.osu_binaries.stagedir,
                                       self.osu_binaries.build_prefix,
                                       'c',
                                       bench_path)
        if build_type == 'rocm':
            self.executable = os.path.join(self.osu_binaries.stagedir, self.osu_binaries.build_prefix,
                                           'map_rank_to_gpu ') + self.executable

@rfm.simple_test
class lumi_osu_pt2pt_check(lumi_osu_benchmarks):
    valid_systems = ['lumi:small', 'lumi:gpu'] 
    use_multithreading = False
    time_limit = '10m'
    benchmark_info = parameter([
        ('mpi.pt2pt.osu_bw', 'bandwidth'),
        ('mpi.pt2pt.osu_latency', 'latency')
    ], fmt=lambda x: x[0], loggable=True)
    osu_binaries = fixture(lumi_build_osu_benchmarks, scope='environment')
    allref = {
        'mpi.pt2pt.osu_bw': {
            'cpu': {
                'lumi:small': {
                    'bandwidth': (6757.0, -0.10, None, 'MB/s')
                }
            },
            'rocm': {
                'lumi:gpu': {
                    'bandwidth': (4850.0, -0.10, None, 'MB/s')
                }
            }
        },
        'mpi.pt2pt.osu_latency': {
            'cpu': {
                'lumi:small': {
                    'latency': (1.8, None, 0.80, 'us')
                }
            },
            'rocm': {
                'lumi:gpu': {
                    'latency': (3.75, None, 0.50, 'us')
                }
            }
        }
    }

    @run_after('init')
    def setup_per_build_type(self):
        build_type = self.osu_binaries.build_type
        if build_type == 'rocm':
            self.valid_systems = ['lumi:gpu']
            self.device_buffers = 'rocm'
            self.num_gpus_per_node = 1
            self.executable_opts = ['-c', '-d', 'rocm', 'D', 'D']
            self.valid_prog_environs = ['builtin-hip']
            self.variables = {'MPICH_GPU_SUPPORT_ENABLED': '1'} 
        else:
            self.valid_systems = ['lumi:small']
            self.valid_prog_environs = ['cpeGNU', 'cpeCray']
            self.executable_opts = ['-c']

        with contextlib.suppress(KeyError):
            self.reference = self.allref[self.benchmark_info[0]][build_type]

    @run_before('run')
    def add_exec_prefix(self):
        build_type = self.osu_binaries.build_type
        bench_path = self.benchmark_info[0].replace('.', '/')
        # update directory structure
        self.executable = os.path.join(self.osu_binaries.stagedir,
                                       self.osu_binaries.build_prefix,
                                       'c',
                                       bench_path)
        if build_type == 'rocm':
            self.executable = os.path.join(self.osu_binaries.stagedir, self.osu_binaries.build_prefix,
                                           'map_rank_to_gpu ') + self.executable

    @sanity_function
    def validate_test(self):
        # with validation
        return sn.assert_found(rf'^{self.message_size}.*Pass', self.stdout)


@rfm.simple_test
class lumi_osu_collective_check(lumi_osu_benchmarks):
    benchmark_info = parameter([
        #('mpi.collective.osu_alltoall', 'latency'),
        ('mpi.collective.osu_allreduce', 'latency'),
    ], fmt=lambda x: x[0], loggable=True)
    num_nodes = parameter([1, 2])
    use_multithreading = False
    valid_systems = ['lumi:small', 'lumi:gpu'] 
    valid_prog_environs = ['cpeGNU', 'cpeCray', 'builtin-hip']
    osu_binaries = fixture(lumi_build_osu_benchmarks, scope='environment')
    allref = {
        'mpi.collective.osu_allreduce': {
            1: {
                'lumi:small': {
                    'latency': (10.0, None, 0.1, 'us')
                }
            },
            2: {
                'lumi:small': {
                    'latency': (7.25, None, 0.05, 'us')
                }
            },
        },
    }

    @run_after('init')
    def setup_by_scale(self):
        with contextlib.suppress(KeyError):
            self.reference = self.allref[self.num_nodes]

    @run_after('init')
    def setup_per_build_type(self):
        build_type = self.osu_binaries.build_type
        if build_type == 'rocm':
            self.valid_systems = ['lumi:gpu']
            self.device_buffers = 'rocm'
            self.num_gpus_per_node = 8
            self.num_tasks_per_node = 8
            self.num_tasks = self.num_tasks_per_node*self.num_nodes
            self.executable_opts = ['-d', 'rocm', 'D', 'D']
            self.valid_prog_environs = ['builtin-hip']
            self.variables = {'MPICH_GPU_SUPPORT_ENABLED': '1'} 
        else:
            self.valid_systems = ['lumi:small']
            self.valid_prog_environs = ['cpeGNU', 'cpeCray']
            self.num_tasks_per_node = 128
            self.num_tasks = self.num_tasks_per_node*self.num_nodes
            self.executable_opts = ['-c']
