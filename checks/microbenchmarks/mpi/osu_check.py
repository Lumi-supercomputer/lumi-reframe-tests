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
   version = '7.3'
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
    osu_benchmarks = fixture(lumi_fetch_osu_benchmarks, scope='session', variables={'version': '7.3'})

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
    use_multithreading = False
    exclusive_access = True
    time_limit = '10m'
    tags = {'production', 'craype'}
    maintainers = ['@mszpindler']

    perf_relative = variable(float, value=0.0, loggable=True)

    @run_after('init')
    def setup_per_build_type(self):
        build_type = self.osu_binaries.build_type
        if build_type == 'rocm':
            self.device_buffers = 'rocm'
            self.valid_systems = ['lumi:gpu']
            self.valid_prog_environs = ['PrgEnv-amd']
            self.modules = ['libfabric', 'rocm']
            self.env_vars = {'MPICH_GPU_SUPPORT_ENABLED': '1'} 
        else:
            self.valid_systems = ['lumi:cpu']
            self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-cray']

    @run_before('run')
    def add_exec_prefix(self):
        bench_path = self.benchmark_info[0].replace('.', '/') 
        # update directory structure
        self.executable = os.path.join(self.osu_binaries.stagedir,
                                       self.osu_binaries.build_prefix,
                                       'c', 
                                       bench_path)

    @run_after('performance')
    def set_relative_perf(self):
        var_name = self.benchmark_info[1]
        key_str = self.current_partition.fullname+':'+var_name
        try:
            found = self.perfvalues[key_str]
        except KeyError:
            return None

        if var_name == 'latency':
            if self.perfvalues[key_str][1] != 0:
                self.perf_relative = ((self.perfvalues[key_str][1]-self.perfvalues[key_str][0])/self.perfvalues[key_str][1])
        if var_name == 'bandwidth':
            if self.perfvalues[key_str][1] != 0:
                self.perf_relative = ((self.perfvalues[key_str][0]-self.perfvalues[key_str][1])/self.perfvalues[key_str][1])

@rfm.simple_test
class lumi_osu_pt2pt_check(lumi_osu_benchmarks):
    # third naming level has been added in versions 7.x of OSU microbenchmarks
    benchmark_info = parameter([
        ('mpi.pt2pt.standard.osu_bw', 'bandwidth'),
        ('mpi.pt2pt.standard.osu_mbw_mr', 'bandwidth'),
        ('mpi.pt2pt.standard.osu_bibw', 'bandwidth'),
        ('mpi.pt2pt.standard.osu_latency', 'latency'),
        ('mpi.pt2pt.standard.osu_multi_lat', 'latency')
    ], fmt=lambda x: x[0], loggable=True)
    osu_binaries = fixture(lumi_build_osu_benchmarks, scope='environment')
    allref = {
        'mpi.pt2pt.standard.osu_mbw_mr' : {
            'cpu': {
                'lumi:cpu': {
                    'bandwidth': (29500.0, -0.05, 0.05, 'MB/s')
                }
            },
	    'rocm': {
                'lumi:gpu': {
                    'bandwidth': (95000.0, -0.01, 0.01, 'MB/s')
                }
            }
        },
        'mpi.pt2pt.standard.osu_bw': {
            'cpu': {
                'lumi:cpu': {
                    'bandwidth': (24044.0, -0.05, 0.05, 'MB/s')
                }
            },
            'rocm': {
                'lumi:gpu': {
                    'bandwidth': (23820.0, -0.05, 0.05, 'MB/s')
                }
            }
        },
        'mpi.pt2pt.standard.osu_latency': {
            'cpu': {
                'lumi:cpu': {
                    'latency': (1.8, -0.2, 0.2, 'us')
                }
            },
            'rocm': {
                'lumi:gpu': {
                    'latency': (2.5, -0.1, 0.1, 'us')
                }
            }
        },
        'mpi.pt2pt.standard.osu_multi_lat': {
            'cpu': {
                'lumi:cpu': {
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
    def setup_refs(self):
        build_type = self.osu_binaries.build_type
        bench_type = self.benchmark_info[1]
        if bench_type == 'bandwidth':
            self.num_iters = 100
        with contextlib.suppress(KeyError):
            self.reference = self.allref[self.benchmark_info[0]][build_type]


    @run_before('run')
    def setup_num_tasks(self):
        bench_name = self.benchmark_info[0]
        build_type = self.osu_binaries.build_type
        if bench_name == 'mpi.pt2pt.standard.osu_mbw_mr' or bench_name == 'mpi.pt2pt.standard.osu_multi_lat': 
            self.num_tasks_per_node = 8
            self.job.launcher.options = ['--cpu-bind="map_cpu:1,14,17,30,33,46,49,62"']
        else:
            self.num_tasks_per_node = 1
        self.num_tasks = 2*self.num_tasks_per_node
        if build_type == 'rocm':
            self.num_gpus_per_node = self.num_tasks_per_node
            # update map_rank_to_gpu wrapper
            if self.num_gpus_per_node > 1:
                self.executable = './map_rank_to_gpu ' + self.executable

    @sanity_function
    def validate_test(self):
        # with validation
        return sn.assert_found(rf'^{self.message_size}.*Pass', self.stdout)

@rfm.simple_test
class lumi_osu_collective_check(lumi_osu_benchmarks):
    valid_systems = ['lumi:cpu', 'lumi:gpu'] 
    use_multithreading = False
    # third naming level has been added in versions 7.x of OSU microbenchmarks
    benchmark_info = parameter([
        ('mpi.collective.blocking.osu_allreduce', 'latency'),
    ], fmt=lambda x: x[0], loggable=True)
    num_nodes = parameter([1, 2, 4])
    osu_binaries = fixture(lumi_build_osu_benchmarks, scope='environment')
    allref = {
        'mpi.collective.blocking.osu_allreduce': {
            1: {
                'cpu': {
                    'lumi:cpu': {
                        'latency': (4.85, -0.2, 0.2, 'us')
                    },
                },
                'rocm': {
                    'lumi:gpu': {
                        'latency': (1.25, -0.2, 0.2, 'us')
                    },
                },
            },
            2: {
                'cpu': {
                    'lumi:cpu': {
                        'latency': (7.5, -0.2, 0.2, 'us')
                    },
                },
                'rocm': {
                    'lumi:gpu': {
                        'latency': (6.0, -0.2, 0.2, 'us')
                    },
                },
            },
            4: {
                'cpu': {
                    'lumi:cpu': {
                        'latency': (10.5, -0.25, 0.25, 'us')
                    },
                },
                'rocm': {
                    'lumi:gpu': {
                        'latency': (10.5, -0.25, 0.25, 'us')
                    },
                },
            },
        },
    }

    @run_before('run')
    def setup_num_tasks(self):
        build_type = self.osu_binaries.build_type
        if build_type == 'rocm':
            self.num_tasks_per_node = 8
            self.num_gpus_per_node = self.num_tasks_per_node
            if self.num_gpus_per_node > 1:
                self.executable = './map_rank_to_gpu ' + self.executable
        else:
            self.num_tasks_per_node = 128
        self.num_tasks = self.num_tasks_per_node*self.num_nodes

    @run_after('init')
    def setup_refs(self):
        build_type = self.osu_binaries.build_type
        with contextlib.suppress(KeyError):
            self.reference = self.allref[self.benchmark_info[0]][self.num_nodes][build_type]

