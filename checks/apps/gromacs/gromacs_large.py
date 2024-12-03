import os
import reframe as rfm
import reframe.utility.sanity as sn

@rfm.simple_test
class lumi_gromacs_large(rfm.RunOnlyRegressionTest):
    '''GROMACS "Grappa" benchmark.
    Páll, S., & Alekseenko, A. (2024). Supplementary information for "GROMACS on AMD GPU-Based HPC Platforms: Using SYCL for Performance and Portability" [Data set]. 
    [https://doi.org/10.5281/zenodo.11087335](https://zenodo.org/doi/10.5281/zenodo.11087334)
    Direct access to the data set: https://zenodo.org/records/11087335/files/grappa-46M.tar.bz2

    The test uses ethanol-46M_RF system to scale on multiple GPU nodes.
    '''
    benchmark_info = {
        'name': 'ethanol',
        'energy_step0_ref': -4.81111e+08,
        'energy_step0_tol': 0.001,
    }

    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['cpeAMD']

    release_environ = parameter(['production', 'leading']) 

    maintainers = ['mszpindler']
    use_multithreading = False
    exclusive_access = True
    num_nodes = parameter([2,4], loggable=True)
    num_gpus_per_node = 8
    time_limit = '10m'
    nb_impl = parameter(['gpu'])

    executable = 'gmx_mpi mdrun'
    tags = {'benchmark', 'contrib', 'gpu'}
    keep_files = ['md.log']

    allref = {
        2: (6.5, -0.05, None, 'ns/day'),
        4: (13.3, -0.05, None, 'ns/day'),
    }

    @run_after('init')
    def set_module_environ(self):
        match self.release_environ:
            case 'production':
                self.modules = ['GROMACS/2024.3-cpeAMD-24.03-rocm', 'rocm/6.0.3', 'AdaptiveCpp/24.06']
                self.tags = {'benchmark', 'production', 'contrib', 'gpu'}
            case 'leading':
                self.modules = ['GROMACS/2024.4-cpeAMD-24.03-rocm', 'rocm/6.2.2', 'AdaptiveCpp/24.06']
                self.tags = {'benchmark', 'testing', 'contrib', 'gpu'}

    @run_after('init')
    def prepare_test(self):
        self.descr = f"GROMACS {self.benchmark_info['name']} GPU benchmark"
        bench_file_path = os.path.join(self.current_system.resourcesdir, 
                                      'datasets',
                                      'gromacs',
                                       self.benchmark_info['name'],
                                      'topol.tpr')
        self.prerun_cmds = [
            f'ln -s {bench_file_path} benchmark.tpr'
        ]

    @run_after('init')
    def apply_module_ver(self):
        #module = f'GROMACS/{self.module_ver}'
        self.modules = ['GROMACS/2024.3-cpeAMD-24.03-rocm', 'rocm/6.0.3']

    @run_after('init')
    def setup_runtime(self):
        self.num_tasks_per_node = 8
        self.num_tasks = self.num_tasks_per_node*self.num_nodes

        self.executable_opts += [
            '-nsteps -1',
            '-maxh 0.15',
            '-nstlist 200',
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
            'GMX_ENABLE_DIRECT_GPU_COMM': '1',
            'GMX_FORCE_GPU_AWARE_MPI': '1',
            'GMX_GPU_PME_DECOMPOSITION': '1',
            'GMX_PMEONEDD': '1',
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
    def verlet_buff_tol(self):
        return sn.extractsingle(r'\s+verlet-buffer-tolerance\s+\=\s+(\S+)', 'md.log', 1, float)

    @sanity_function
    def assert_energy_readout(self):
        return sn.all([
            sn.assert_found('Finished mdrun', 'md.log', 'Run failed to complete'), 
            sn.assert_reference(
                self.energy_step0(),
                self.benchmark_info['energy_step0_ref'],
                -self.benchmark_info['energy_step0_tol'],
                self.benchmark_info['energy_step0_tol']
            ),
            sn.assert_lt(
                self.energy_drift(),
                2*self.verlet_buff_tol()
           ),
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
    tags = {'benchmark', 'gpu'}

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
