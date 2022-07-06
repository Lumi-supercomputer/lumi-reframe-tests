import contextlib
import reframe as rfm
from hpctestlib.microbenchmarks.mpi.osu import (build_osu_benchmarks,
                                                osu_build_run)

class lumi_build_osu_benchmarks(build_osu_benchmarks):
    build_type = parameter(['cpu', 'rocm'])

    @run_after('init')
    def setup_modules(self):
        if self.build_type == 'cpu':
            self.valid_systems = ['lumi:small']
        elif self.build_type == 'rocm':
            self.build_locally = False
            self.valid_systems = ['lumi:eap']
            self.modules = ['rocm']


class lumi_osu_benchmarks(osu_build_run):
    tags = {'production', 'benchmark',}
    maintainers = ['@rsarm', '@mszpindler']


@rfm.simple_test
class lumi_osu_pt2pt_check(lumi_osu_benchmarks):
    valid_systems = ['lumi:small', 'lumi:eap'] 
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
                'lumi:eap': {
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
                'lumi:eap': {
                    'latency': (3.75, None, 0.50, 'us')
                }
            }
        }
    }

    @run_after('init')
    def setup_per_build_type(self):
        build_type = self.osu_binaries.build_type
        if build_type == 'rocm':
            self.device_buffers = 'rocm'
            self.num_gpus_per_node = 1
            self.executable_opts = ['-d', 'cuda', 'D', 'D']
            self.valid_systems = ['lumi']
            self.valid_prog_environs = ['builtin-hip']
            self.variables = {'MPICH_GPU_SUPPORT_ENABLED': '1'} 
        else:
            self.valid_systems = ['lumi:small']
            self.valid_prog_environs = ['cpeGNU', 'cpeCray']

        with contextlib.suppress(KeyError):
            self.reference = self.allref[self.benchmark_info[0]][build_type]


@rfm.simple_test
class lumi_osu_collective_check(lumi_osu_benchmarks):
    benchmark_info = parameter([
        ('mpi.collective.osu_alltoall', 'latency'),
        ('mpi.collective.osu_allreduce', 'latency'),
    ], fmt=lambda x: x[0], loggable=True)
    num_nodes = parameter([2, 4])
    valid_systems = ['lumi']
    valid_prog_environs = ['cpeGNU', 'cpeCray']
    osu_binaries = fixture(lumi_build_osu_benchmarks, scope='environment')
    allref = {
        'mpi.collective.osu_allreduce': {
            4: {
                'lumi:small': {
                    'latency': (10.0, None, 0.1, 'us')
                }
            },
            2: {
                'lumi:small': {
                    'latency': (7.25, None, 0.05, 'us')
                }
            }
        },
        'mpi.collective.osu_alltoall': {
            4: {
                'lumi:small': {
                    'latency': (6.0, None, 0.1, 'us')
                }
            },
            2: {
                'lumi:small': {
                    'latency': (2.0, None, 0.05, 'us')
                }
            }
        }
    }

    @run_after('init')
    def setup_by_scale(self):
        self.num_tasks = self.num_nodes

        with contextlib.suppress(KeyError):
            self.reference = self.allref[self.num_nodes]
