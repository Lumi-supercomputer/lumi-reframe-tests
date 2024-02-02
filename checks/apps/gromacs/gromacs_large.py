import os
import reframe as rfm
import reframe.utility.sanity as sn
from hpctestlib.sciapps.gromacs.benchmarks import gromacs_check

# This is based on CSCS Reframe GROMACS tests library.
# The test uses ethanol-46M_RF system to scale on multiple GPU node.


#util.find_modules('GROMACS')

@rfm.simple_test
class lumi_gromacs_large(gromacs_check):
    # benchmark_info parameter encodes: 
    #       name,
    #       reference values for: total en. at step 0, conserved en. drift, 
    #       error tolerance for: total en. at step 0, conserved en. drift,
    benchmark_info = parameter([
        ('ethanol', 
         [-4.81111e+08, 4.20e-04], 
         [0.001, 0.25]
        ), 
    ], fmt=lambda x: x[0], loggable=True)
    nb_impl = parameter(['gpu'])
    num_nodes = parameter([2,4], loggable=True)

    use_multithreading = False
    exclusive_access = True
    num_gpus_per_node = 8
    time_limit = '10m'

    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['cpeAMD']
    modules = ['GROMACS/2023.3-cpeAMD-22.12-HeFFTe-GPU']

    maintainers = ['mszpindler']
    tags = {'benchmark', 'contrib/22.12'}

    allref = {
        2: (6.5, -0.05, None, 'ns/day'),
        4: (13.3, -0.05, None, 'ns/day'),
    }

    @loggable
    @property
    def bench_name(self):
        '''The benchmark name.

        :type: :class:`str`
        '''

        return self.__bench

    @property
    def energy_step0_ref(self):
        '''The energy reference value for this benchmark.

        :type: :class:`str`
        '''
        return self.__nrg_ref[0]

    @property
    def energy_drift_ref(self):
        '''The energy drift reference value for this benchmark.

        :type: :class:`str`
        '''
        return self.__nrg_ref[1]

    @property
    def energy_step0_tol(self):
        return self.__nrg_tol[0]

    @property
    def energy_drift_tol(self):
        return self.__nrg_tol[1]


    @run_after('init')
    def prepare_test(self):
        self.__bench, self.__nrg_ref, self.__nrg_tol = self.benchmark_info
        self.descr = f'GROMACS {self.__bench} GPU benchmark (LUMI contrib build {self.modules})' 
        bench_file_path = os.path.join(self.current_system.resourcesdir, 
                                      'gromacs-benchmarks', 
                                       self.__bench, 
                                      'topol.tpr')
        self.prerun_cmds = [
            f'ln -s {bench_file_path} benchmark.tpr'
        ]

    @run_after('init')
    def setup_run_opts(self):
        self.num_tasks_per_node = 8
        self.num_tasks = self.num_tasks_per_node*self.num_nodes

        self.executable_opts += [
            '-nsteps 20000',
            '-nstlist 400',
            '-noconfout',
            '-notunepme',
            '-resetstep 10000',
            '-nb', self.nb_impl,
            '-pme', 'cpu', 
            '-update', 'gpu',
            '-bonded', 'gpu',
            '-s benchmark.tpr'
        ]
        self.env_vars = {
            'MPICH_GPU_SUPPORT_ENABLED': '1',
            'OMP_NUM_THREADS': '7',
            'OMP_PROC_BIND': 'close',
            'OMP_PLACES': 'cores',
            #'OMP_DISPLAY_ENV': '1',
            #'OMP_DISPLAY_AFFINITY': 'TRUE',
            'GMX_ENABLE_DIRECT_GPU_COMM': '1',
            'GMX_FORCE_GPU_AWARE_MPI': '1',
            'GMX_GPU_PME_DECOMPOSITION': '1',
            'GMX_PMEONEDD': '1'
        }

    @run_before('run')
    def set_cpu_mask(self):
        cpu_bind_mask = '0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000'
        self.job.launcher.options = [f'--cpu-bind=mask_cpu:{cpu_bind_mask}']

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

    @performance_function('ns/day')
    def perf(self):
        return sn.extractsingle(r'Performance:\s+(?P<perf>\S+)',
                                'md.log', 'perf', float)

    @deferrable
    def energy_step0(self):
        #LJ (SR)   Coulomb (SR)      Potential    Kinetic En.   Total Energy
        return sn.extractsingle(r'\s+LJ \(SR\)\s+\s+Coulomb \(SR\)\s+Potential\s+Kinetic En\.\s+Total Energy\n'
                                r'(\s+\S+){4}\s+(?P<energy>\S+)',
                                'md.log', 'energy', float)

    @deferrable
    def energy_drift(self):
        return sn.extractsingle(r'\s+Conserved\s+energy\s+drift\:\s+(\S+)', 'md.log', 1, float)
    
    @deferrable
    def assert_energy_drift(self):
        if self.num_nodes == 2:
            return True
        else:
            return sn.assert_reference(self.energy_drift(), self.energy_drift_ref, -self.energy_drift_tol, self.energy_drift_tol) #'Failed to meet reference value for conserved energy drift'

    @sanity_function 
    def assert_run_correct(self):
        return sn.all([
            sn.assert_found('Finished mdrun', 'md.log', 'Run failed to complete'), 
            sn.assert_reference(self.energy_step0(), self.energy_step0_ref, -self.energy_step0_tol, self.energy_step0_tol), 
            self.num_nodes == 2 or sn.assert_reference(self.energy_drift(), self.energy_drift_ref, -self.energy_drift_tol, self.energy_drift_tol), 
        ])

    @run_before('run')
    def setup_run(self):
        try:
            found = self.allref[self.num_nodes]
        except KeyError:
            self.skip(f'Configuration with {self.num_nodes} node(s) of '
                      f'{self.bench_name!r} is not supported on {arch!r}')

        # Setup performance references
        self.reference = {
            '*': {
                'perf': self.allref[self.num_nodes]
            }
        }

@rfm.simple_test
class lumi_gromacs_scaling(lumi_gromacs_large):
    num_nodes = parameter([2**n for n in range(1,9)], loggable=True)

    maintainers = ['mszpindler']
    tags = {'benchmark', 'contrib/22.12'}

    allref = {
        2: (6.5, -0.05, None, 'ns/day'),
        4: (13.3, -0.05, None, 'ns/day'),
        8: (25.8, -0.05, None, 'ns/day'),
        16: (45.5, -0.05, None, 'ns/day'),
        32: (68.0, -0.05, None, 'ns/day'),
        64: (94.1, -0.05, None, 'ns/day'),
        128: (112.1, -0.05, None, 'ns/day'),
        256: (13.3, -0.05, None, 'ns/day'),
    }
