import reframe as rfm
import reframe.utility.sanity as sn


class abinit_check(rfm.RunOnlyRegressionTest):
    modules = ['ABINIT-9.6.2']
    executable = 'abinit'
    maintainers = ['mszpindler']
    strict_check = False

    @run_after('init')
    def set_input(self):
        self.prerun_cmds = [
                #f'curl -LJO https://raw.githubusercontent.com/abinit/abinit/master/tests/paral/Input/t01.abi',
                f'sed -i -e "/nstep/s/2/20/" t01.abi',
                #f'curl -LJO https://raw.githubusercontent.com/abinit/abinit/master/tests/Psps_for_tests/14si.psp'
            ]
        self.executable_opts = ['t01.abi']
        self.variables = {
                'ABI_PSPDIR': '.',
        }

    @run_after('init')
    def set_prgenv(self):
        if self.current_system.name in ['lumi']:
            self.valid_prog_environs = ['cpeGNU']
        else:
            self.valid_prog_environs = ['builtin']

    @sanity_function
    def assert_simulation_success(self):
        return sn.assert_found(r'Calculation completed',self.stdout)

    @performance_function('s')
    def time(self):
        return sn.extractsingle(r'overall_wall_time:\s+(?P<wtime>\S+)'
                                self.stdout, 'wtime', float)


@rfm.simple_test
class lumi_abinit_cpu_check(abinit_check):
    scale = parameter(['small'])
    valid_systems = ['lumi']
    refs_by_scale = {
        'small': {
            'lumi:cpu': {'time': (10.0, None, 5, 's')}, 
        },
        'large': {
            #
        }
    }

    @run_after('init')
    def setup_by_scale(self):
        self.descr = f'Abinit CPU check (version: {self.scale})'
        self.tags |= {'maintenance', 'production'}
        if self.scale == 'small':
            if self.current_system.name in ['lumi']:
                self.num_tasks = 10

        self.reference = self.refs_by_scale[self.scale]


# Fix it: GPU enabled Abinit instance (module)
#@rfm.simple_test
#class lumi_abinit_gpu_check(abinit_check):
#    scale = parameter(['small'])
#    valid_systems = ['lumi']
#    refs_by_scale = {
#        'small': {
#            'lumi:gpu': {'time': (182.0, None, 0.05, 's')}, # Fix it
#        },
#        'large': {
#            #
#        }
#    }
#
#    @run_after('init')
#    def setup_by_scale(self):
#        self.descr = f'Abinit GPU check (version: {self.scale})'
#        if self.scale == 'small':
#            if self.current_system.name in ['lumi']:
#                self.num_tasks = 24
#                self.num_tasks_per_node = 24
#                self.num_gpus_per_node = 1
#                #self.num_cpus_per_task = 2
#                #self.variables = {
#                #    'OMP_NUM_THREADS': str(self.num_cpus_per_task)
#                #}
#
#        self.reference = self.refs_by_scale[self.scale]
