import reframe as rfm
import reframe.utility.sanity as sn


class cp2k_check(rfm.RunOnlyRegressionTest):
    executable = 'cp2k.psmp'
    maintainers = ['mszpindler']
    executable_opts = ['H2O-256.inp']

    @sanity_function
    def assert_energy_diff(self):
        energy = sn.extractsingle(
            r'\s+ENERGY\| Total FORCE_EVAL \( QS \) '
            r'energy [\[\(]a\.u\.[\]\)]:\s+(?P<energy>\S+)',
            self.stdout, 'energy', float, item=-1
        )
        energy_reference = -4404.2323
        energy_diff = sn.abs(energy-energy_reference)
        return sn.all([
            sn.assert_found(r'PROGRAM STOPPED IN', self.stdout),
            sn.assert_eq(sn.count(sn.extractall(
                r'(?i)(?P<step_count>STEP NUMBER)',
                self.stdout, 'step_count')), 10),
            sn.assert_lt(energy_diff, 1e-4)
        ])

    @performance_function('s')
    def time(self):
        return sn.extractsingle(r'^ CP2K(\s+[\d\.]+){4}\s+(?P<perf>\S+)',
                                self.stdout, 'perf', float)


@rfm.simple_test
class lumi_cp2k_cpu_check(cp2k_check):
    modules = ['CP2K']
    valid_systems = ['lumi:small']
    valid_prog_environs = ['cpeGNU']
    descr = f'CP2K CPU check'
    reference = {
        'lumi:small': {'time': (152.644, None, 0.05, 's')},
    }
    num_tasks = 256
    num_tasks_per_node = 128

    tags = {'contrib/22.08', 'contrib/22.12'}


@rfm.simple_test
class lumi_cp2k_gpu_check(cp2k_check):
    #modules = ['CP2K/2024.2-cpeGNU-24.03-rocm']
    modules = ['CP2K']
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['cpeAMD']
    descr = f'CP2K CPU check'

    # TODO: update the timings with more usefull ones.
    reference = {
    }
    num_tasks = 16
    num_tasks_per_node = 8

    # CP2K seems to have problems if a rank has more than one GPU.
    #  Furthermore, there is also this: https://confluence.cscs.ch/spaces/KB/pages/868823032/Known+MPI+issues#KnownMPIissues-%22cxil_map%3Awriteerror%22whendoinginter-nodeGPU-awareMPIcommunication
    extra_resources = {
            "_rfm_gpu": {"num_gpus_per_task": 1},
    }

    tags = {'contrib/22.08', 'contrib/22.12'}

