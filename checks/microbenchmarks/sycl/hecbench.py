# This check is using code from https://github.com/zjin-lcf/HeCBench
import reframe as rfm
import reframe.utility.sanity as sn

@rfm.simple_test
class HeCBench_heat(rfm.RegressionTest):
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['cpeGNU', 'cpeAMD']
    modules = ['AdaptiveCpp']
    num_gpus_per_node = 1
    build_system = 'Make'

    maintainers = ['mszpindler']
    tags = {'contrib/23.09'}

    reference = {
        'lumi:gpu': {
            'Time':  (1.0, -0.2, 0.2, 's'),
            'Bandwidth': (1090.0, -0.05, 0.05, 'GB/s'),
        }
    }

    @run_before('compile')
    def set_make_flags(self):
         self.prebuild_cmds = [
             'curl -LJO https://raw.githubusercontent.com/zjin-lcf/HeCBench/master/src/heat-sycl/heat.cpp',
             'curl -LJO https://raw.githubusercontent.com/zjin-lcf/HeCBench/master/src/heat-sycl/Makefile',
         ]
         self.build_system.flags_from_environ= False
         self.build_system.options = ['CC=acpp HIP=yes HIP_ARCH=gfx90a CFLAGS="-std=c++17 -Wall --acpp-targets=hip:gfx90a -O3 -DUSE_GPU"']

    @run_before('run')
    def set_executable(self):
         self.executable = './heat'
         self.executable_opts = ['4096 1000']

    @sanity_function
    def check_norm(self):
        error_norm = sn.extractsingle(r'Error\s+\(L2norm\)\:\s+(\S+)', self.stdout, 1, float)
        return sn.assert_lt(error_norm, 1e-9)

    @performance_function('GB/s')
    def bandwidth(self):
        return sn.extractsingle(r'^Bandwidth\s+\(GB/s\)\:\s+(\S+)', self.stdout, 1, float)

    @performance_function('s')
    def total_time(self):
        return sn.extractsingle(r'^Total\s+time\s+\(s\)\:\s+(\S+)', self.stdout, 1, float)


    @run_before('performance')
    def set_perf_variables(self):

        self.perf_variables = {
            'Time': self.total_time(),
            'Bandwidth': self.bandwidth(),
        }

