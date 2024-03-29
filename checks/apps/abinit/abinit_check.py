import reframe as rfm
import reframe.utility.sanity as sn


class abinit_check(rfm.RunOnlyRegressionTest):
    modules = ['ABINIT']
    executable = 'abinit'
    maintainers = ['mszpindler']
    prerun_cmds = ['sed -i -e "/nstep/s/2/20/" t01.abi']
    executable_opts = ['t01.abi']
    env_vars = {
        'ABI_PSPDIR': '.',
    }


    @sanity_function
    def assert_simulation_success(self):
        return sn.assert_found(r'Calculation completed',self.stdout)

    @performance_function('s')
    def time(self):
        return sn.extractsingle(r'overall_wall_time:\s+(?P<wtime>\S+)',
                                self.stdout, 'wtime', float)


@rfm.simple_test
class lumi_abinit_cpu_check(abinit_check):
    valid_systems = ['lumi:small']
    valid_prog_environs = ['cpeGNU']
    descr = f'Abinit CPU check'
    num_tasks = 10
    reference = {
        'lumi:small': {'time': (10.0, None, 0.5, 's')}, 
    } 

    tags = {'contrib/21.12'}
