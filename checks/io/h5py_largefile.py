import reframe as rfm
import reframe.utility.sanity as sn

@rfm.simple_test
class h5_LargeFile(rfm.RunOnlyRegressionTest):
    valid_systems = ['lumi:gpu', 'lumi:login']
    valid_prog_environs = ['cpeGNU']
    modules = ['cray-python', 'h5py']  
    executable = 'python'
    executable_opts = ['h5_large_file.py']
    num_tasks = 1

    @sanity_function
    def validate_write_time(self):
        return sn.assert_found(r'write time+\s*', self.stdout)
