import os
import reframe as rfm
import reframe.utility.sanity as sn


class osu_gpu_pt2pt_bw_base(rfm.RunOnlyRegressionTest):
    # This check was adapted to GPUs from mpi4py's script
    # https://github.com/mpi4py/mpi4py/blob/master/demo/osu_bw.py
    descr = 'OSU GPU to GPU bandwith test with mpi4py and cupy'
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['cpeGNU']
    modules = ['rocm', 'CuPy']
    num_tasks = 2
    exclusive_access = True
    executable = 'python osu_bw_cupy.py'
    env_vars = {
        'MPICH_GPU_SUPPORT_ENABLED': '1',
        'LD_PRELOAD': '${CRAY_MPICH_ROOTDIR}/gtl/lib/libmpi_gtl_hsa.so'
    }

    tags = {'python', 'lumi-stack'}

    @sanity_function
    def assert_found_max_bandwidth(self):
        max_bandwidth = r'4194304'
        return sn.assert_found(max_bandwidth, self.stdout)

    @performance_function('MB/s')
    def bandwidth(self):
        return sn.extractsingle(r'4194304\s+(?P<bw>\S+)',
                                self.stdout, 'bw', float)


@rfm.simple_test
class osu_gpu_pt2pt_bw_two_nodes_test(osu_gpu_pt2pt_bw_base):
    num_tasks_per_node = 1
    num_gpus_per_node = 1
    reference = {
        'lumi:gpu': {'bandwidth': (23952.10, -0.05, None, 'MB/s')}
    }


@rfm.simple_test
class osu_gpu_pt2pt_bw_single_node_test(osu_gpu_pt2pt_bw_base):
    num_tasks_per_node = 2
    num_gpus_per_node = 2
    reference = {
        'lumi:gpu': {'bandwidth': (125288.14, -0.05, None, 'MB/s')}
    }

    @run_before('run')
    def set_gpu_binding(self):
        self.job.options = ['--gpu-bind=closest']
