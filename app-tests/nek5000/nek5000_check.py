import reframe as rfm
import reframe.utility.sanity as sn

@rfm.simple_test
class nek5000_check(rfm.RegressionTest):
    valid_systems = ['lumi']
    valid_prog_environs = ['cpeGNU']
    modules = ['partition/C', 'Nek5000']
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
        return sn.extractsingle(r'total elapsed time \s+ \: \s+(?P<wtime>\S+) sec', 
                                self.stdout, 'wtime', float)
