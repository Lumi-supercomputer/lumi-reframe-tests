import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class HipInfo(rfm.RegressionTest):
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin-hip']
    modules = ['rocm']
    build_system = 'SingleSource'
    sourcepath = 'hipInfo.cpp'
    executable = 'hipInfo'
    build_locally = False
    num_gpus_per_node = 8

    @run_before('compile')
    def set_compiler_flags(self):
        self.build_system.cc = 'hipcc'
        self.build_system.cxx = 'hipcc'
        self.gpu_build = 'hip'

    @sanity_function
    def validate_solution(self):
        num_devices = sn.count(sn.findall(r'^device#', self.stdout))
        return sn.assert_eq(num_devices, self.num_gpus_per_node)
