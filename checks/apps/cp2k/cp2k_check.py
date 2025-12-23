import reframe as rfm
import reframe.utility.sanity as sn


class cp2k_check(rfm.RunOnlyRegressionTest):
    maintainers = ['philip-paul-mueller']

    perf_relative = variable(float, value=0.0, loggable=True)

    # TODO: Make sure that the GPU timings are meaningfull.
    reference = {
        'lumi:cpu': {'time': (152.644, None, 0.05, 's')},
        'lumi:gpu': {'time': (165.0, None, 0.05, 's')},
    }

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

    @run_after('performance')
    def lower_the_better(self):
        perf_var = 'time'
        key_str = self.current_partition.fullname+':'+perf_var
        try:
            found = self.perfvalues[key_str]
        except KeyError:
            return None

        if self.perfvalues[key_str][1] != 0:
            self.perf_relative = ((self.perfvalues[key_str][1]-self.perfvalues[key_str][0])/self.perfvalues[key_str][1])

@rfm.simple_test
class lumi_cp2k_cpu_check(cp2k_check):
    modules = ['CP2K']
    valid_systems = ['lumi:cpu']
    valid_prog_environs = ['cpeGNU']
    descr = f'CP2K CPU check'
    tags = {'contrib', 'performance'}

    num_tasks = 256
    num_tasks_per_node = 128

    executable = 'cp2k.psmp'
    executable_opts = ['H2O-256.inp']


@rfm.simple_test
class lumi_cp2k_gpu_check(cp2k_check):
    """The CP2K GPU test on LUMI.

    The way how CP2K is called is based on the [documentation](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/c/CP2K/#example-batch-scripts)
    """
    modules = ['CP2K/2024.2-cpeGNU-24.03-rocm']
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['cpeGNU']
    descr = 'CP2K GPU check'
    tags = {'contrib', 'performance'}

    num_cpus_per_task = 7
    num_tasks = 16
    num_tasks_per_node = 8
    num_gpus_per_node = 8

    executable = 'cp2k.psmp'
    executable_opts = ['H2O-256.inp']

    # We have to use the script here becuase we have to make sure that every
    #  rank has exactly one GPU. It would be nice to use the `--gpus-per-task`
    #  flag but that does not seem to work.
    @run_after('init')
    def add_select_gpu_wrapper(self):
        self.prerun_cmds += [
            'cat << EOF > select_gpu',
            '#!/bin/bash',
            'export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID',
            'exec \$*',
            'EOF',
            'chmod +x ./select_gpu'
        ]
        self.executable = './select_gpu ' + self.executable
    
    @run_before('run')
    def set_cpu_binding_mask(self):
        self.job.launcher.options = ["--cpu-bind=mask_cpu:7e000000000000,7e00000000000000,7e0000,7e000000,7e,7e00,7e00000000,7e0000000000"]

    prerun_cmds = ["ulimit -s unlimited"]
    env_vars = {
        "MPICH_OFI_NIC_POLICY": "GPU",
        "MPICH_GPU_SUPPORT_ENABLED": "1",
        "OMP_PLACES": "cores",
        "OMP_PROC_BIND": "close",
        "OMP_NUM_THREADS": "${SLURM_CPUS_PER_TASK}",
        "OMP_STACKSIZE": "512M",
    }
