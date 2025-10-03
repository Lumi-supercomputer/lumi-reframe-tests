import os
import reframe as rfm
import reframe.utility.sanity as sn


class mpi4py_osu_pt2pt_bw_base(rfm.RunOnlyRegressionTest):
    # This check was adapted to GPUs from mpi4py's script
    # https://github.com/mpi4py/mpi4py/blob/master/demo/osu_bw.py

    descr = 'OSU CPU and GPU to GPU bandwith test with mpi4py and cupy'
    device_type = parameter(['cpu', 'gpu'])
    num_tasks = 2
    executable = 'python'

    perf_relative = variable(float, value=0.0, loggable=True)

    tags = {'production', 'python', 'craype', 'contrib', 'performance'}

    @run_after('init')
    def setup_test(self):
        if self.device_type == 'gpu':
            self.valid_systems = ['lumi:gpu']
            self.valid_prog_environs = ['cpeGNU']
            self.modules = ['cray-python', 'rocm', 'CuPy']
            self.executable_opts = ['osu_bw_cupy.py']
            self.env_vars = {
                'MPICH_GPU_SUPPORT_ENABLED': '1',
                'LD_PRELOAD': '${CRAY_MPICH_ROOTDIR}/gtl/lib/libmpi_gtl_hsa.so'
            }
        if self.device_type == 'cpu':
            self.valid_systems = ['lumi:small']
            self.valid_prog_environs = ['PrgEnv-gnu']
            self.modules = ['cray-python']
            self.executable_opts = ['osu_bw.py']
   
    @sanity_function
    def assert_found_max_bandwidth(self):
        max_bandwidth = r'4194304'
        return sn.assert_found(max_bandwidth, self.stdout)

    @performance_function('MB/s')
    def bandwidth(self):
        return sn.extractsingle(r'4194304\s+(?P<bw>\S+)',
                                self.stdout, 'bw', float)

    @run_after('performance')
    def higher_the_better(self):
        perf_var = 'bandwidth'
        key_str = self.current_partition.fullname+':'+perf_var
        try:
            found = self.perfvalues[key_str]
        except KeyError:
            return None

        if self.perfvalues[key_str][1] != 0:
            self.perf_relative = ((self.perfvalues[key_str][0]-self.perfvalues[key_str][1])/self.perfvalues[key_str][1])

@rfm.simple_test
class mpi4py_osu_pt2pt_bw_two_nodes_test(mpi4py_osu_pt2pt_bw_base):
    reference = {
        'lumi:gpu':   {'bandwidth': (23952.10, -0.05, None, 'MB/s')},
        'lumi:small': {'bandwidth': (22000.0, -0.05, None, 'MB/s')},
    }

    @run_before('run')
    def setup_job(self):
        self.num_tasks_per_node = 1
        if self.device_type == 'gpu':
            self.num_gpus_per_node = 1
            self.job.options = ['--gpus-per-task=1']
