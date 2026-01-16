import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class AdaptiveCpp_EBCheck(rfm.RegressionTest):
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['cpeGNU']
    executable = 'acpp'
    executable_opts = ['--acpp-version']
    modules = ['EasyBuild-user']
    build_system = 'EasyBuild'

    tags = {'lumi-stack'}

    @run_before('compile')
    def setup_build_system(self):
        self.build_system.easyconfigs = ['AdaptiveCpp-23.10.0-cpeGNU-22.12-rocm-5.2.3.eb']
        self.build_system.options = ['-f --pr-target-account=mszpindler --from-pr 1']

    @run_before('run')
    def prepare_run(self):
        self.modules = self.build_system.generated_modules

    @sanity_function
    def assert_version(self):
        return sn.assert_found(r'\s+AdaptiveCpp version: 23.10.0.*', self.stdout)
