import reframe as rfm
import reframe.utility.sanity as sn

@rfm.simple_test
class nek5000_check(rfm.RegressionTest):
    #modules = ['']
    #user_modules =
    valid_systems = ['lumi']
    valid_prog_environs = ['cpeGNU']
    modules = ['partition/C']
    executable = './nek5000'
    strict_check = False
    descr = (f'Nek500 test')
    maintainers = ['']
    build_system = 'CustomBuild'
    #build_locally = False

    @run_after('init')
    def setup_run(self):
        self.num_tasks = 32

    @run_before('compile')
    def build_test(self):
        self.build_system.commands = [
            f'export PATH=/pfs/lustrep4/users/maciszpin/EasyBuild/SW/LUMI-21.12/C/Nek5000/19.0-cpeGNU-21.12/bin:$PATH', # Replace by application module
            f'export NEK_SOURCE_ROOT=/pfs/lustrep4/users/maciszpin/EasyBuild/SW/LUMI-21.12/C/Nek5000/19.0-cpeGNU-21.12/',
            f'env CC=cc FC=ftn',
            'makenek bp5',
        ]
        self.prebuild_cmds = [
            f'git clone --filter=blob:none --sparse https://github.com/Nek5000/NekExamples.git',
            'cd NekExamples',
            f'git sparse-checkout init --cone',
            f'git sparse-checkout add bp5',
            'cd bp5',
        ]

    @run_before('run')
    def cwd_test(self):
        self.prerun_cmds = [
            'cd NekExamples/bp5',
            'echo bp5 > SESSION.NAME && echo `pwd`"/" >> SESSION.NAME'
        ]

    @sanity_function
    def assert_simulation_success(self):
        return sn.assert_found(r'run successful',self.stdout)

    @performance_function('s')
    def time(self):
        return sn.extractsingle(r'total elapsed time\s+[:]\s+(?P<wtime>\s+\S+)', # Fix scientific notation (E+) conversion
                                self.stdout, 'wtime', float)
