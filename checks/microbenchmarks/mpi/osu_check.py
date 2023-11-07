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

    @run_after('init')
    def setup_per_build_type(self):
        build_type = self.osu_binaries.build_type
        bench_name = self.benchmark_info[0]
        if build_type == 'rocm':
            self.valid_systems = ['lumi:gpu']
            self.valid_prog_environs = ['PrgEnv-amd']
            self.executable_opts = ['-c', '-d', 'rocm', 'D', 'D']
            self.variables = {'MPICH_GPU_SUPPORT_ENABLED': '1'} 
            if bench_name == 'mpi.collective.osu_allreduce':
                self.executable_opts = ['-d', 'rocm', 'D', 'D']
            if bench_name == 'mpi.pt2pt.osu_mbw_mr':
		# Binding for numa-spread mapping
            	self.variables = {'MPICH_GPU_SUPPORT_ENABLED': '1', 'MPICH_GPU_IPC_ENABLED': '1', 'SLURM_CPU_BIND': 'map_cpu:1,17,33,49,14,30,46,62'}
		# Binding for numa-close mapping
            	#self.variables = {'MPICH_GPU_SUPPORT_ENABLED': '1', 'MPICH_GPU_IPC_ENABLED': '1', 'SLURM_CPU_BIND': 'map_cpu:1,14,17,30,33,46,49,62'}
        else:
            self.valid_systems = ['lumi:standard']
            self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-cray']
            self.executable_opts = ['-c']

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
    valid_systems = ['lumi:standard', 'lumi:gpu'] 
    use_multithreading = False
    time_limit = '10m'
    benchmark_info = parameter([
        ('mpi.pt2pt.osu_bw', 'bandwidth'),
        ('mpi.pt2pt.osu_mbw_mr', 'bandwidth'),
        ('mpi.pt2pt.osu_latency', 'latency'),
        ('mpi.pt2pt.osu_multi_lat', 'latency')
    ], fmt=lambda x: x[0], loggable=True)
    osu_binaries = fixture(lumi_build_osu_benchmarks, scope='environment')
    allref = {
        'mpi.pt2pt.osu_mbw_mr' : {
            'cpu': {
                'lumi:standard': {
                    'bandwidth': (120340.0, -0.05, 0.05, 'MB/s')
                }
            },
	    'rocm': {
                'lumi:gpu': {
                    'bandwidth': (482735.0, -0.01, 0.01, 'MB/s')
                }
            }
        },
        'mpi.pt2pt.osu_bw': {
            'cpu': {
                'lumi:standard': {
                    'bandwidth': (24044.0, -0.05, 0.05, 'MB/s')
                }
            },
            'rocm': {
                'lumi:gpu': {
                    'bandwidth': (23820.0, -0.05, 0.05, 'MB/s')
                }
            }
        },
        'mpi.pt2pt.osu_latency': {
            'cpu': {
                'lumi:standard': {
                    'latency': (1.8, -0.2, 0.2, 'us')
                }
            },
            'rocm': {
                'lumi:gpu': {
                    'latency': (2.5, -0.1, 0.1, 'us')
                }
            }
        },
        'mpi.pt2pt.osu_multi_lat': {
            'cpu': {
                'lumi:standard': {
                    'latency': (1.95, -0.2, 0.2, 'us')
                }
            },
            'rocm': {
                'lumi:gpu': {
                    'latency': (3.0, -0.1, 0.1, 'us')
                }
            }
        }
    }

    @run_after('init')
    def setup_num_tasks(self):
        build_type = self.osu_binaries.build_type
        bench_name = self.benchmark_info[0]
        if bench_name == 'mpi.pt2pt.osu_mbw_mr':
            self.num_tasks_per_node = 8
            self.num_tasks = 8
            self.exclusive_access = True
        elif bench_name == 'mpi.pt2pt.osu_multi_lat':
            self.num_tasks_per_node = 8
            self.num_tasks *= self.num_tasks_per_node
        else:
            self.num_tasks_per_node = 1
        if build_type == 'rocm':
            self.num_gpus_per_node = self.num_tasks_per_node

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
            if self.num_gpus_per_node > 1:
            	# update map_rank_to_gpu wrapper
            	self.executable = './map_rank_to_gpu ' + self.executable

    @sanity_function
    def validate_test(self):
        # with validation
        return sn.assert_found(rf'^{self.message_size}.*Pass', self.stdout)


@rfm.simple_test
class lumi_osu_collective_check(lumi_osu_benchmarks):
    valid_systems = ['lumi:standard', 'lumi:gpu'] 
    use_multithreading = False
    benchmark_info = parameter([
        ('mpi.collective.osu_allreduce', 'latency'),
    ], fmt=lambda x: x[0], loggable=True)
    num_nodes = parameter([1, 2, 4])
    osu_binaries = fixture(lumi_build_osu_benchmarks, scope='environment')
    allref = {
        'mpi.collective.osu_allreduce': {
            1: {
                'cpu': {
                    'lumi:standard': {
                        'latency': (4.85, -0.2, 0.2, 'us')
                    },
                },
                'rocm': {
                    'lumi:gpu': {
                        'latency': (2.5, -0.2, 0.2, 'us')
                    },
                },
            },
            2: {
                'cpu': {
                    'lumi:standard': {
                        'latency': (7.5, -0.2, 0.2, 'us')
                    },
                },
                'rocm': {
                    'lumi:gpu': {
                        'latency': (7.95, -0.2, 0.2, 'us')
                    },
                },
            },
            4: {
                'cpu': {
                    'lumi:standard': {
                        'latency': (10.5, -0.25, 0.25, 'us')
                    },
                },
                'rocm': {
                    'lumi:gpu': {
                        'latency': (12.5, -0.1, 0.1, 'us')
                    },
                },
            },
        },
    }

    @run_after('init')
    def setup_num_tasks(self):
        build_type = self.osu_binaries.build_type
        if build_type == 'rocm':
            self.num_tasks_per_node = 8
            self.num_gpus_per_node = self.num_tasks_per_node
        else:
            self.num_tasks_per_node = 128
        self.num_tasks = self.num_tasks_per_node*self.num_nodes

        with contextlib.suppress(KeyError):
            self.reference = self.allref[self.benchmark_info[0]][self.num_nodes][build_type]

