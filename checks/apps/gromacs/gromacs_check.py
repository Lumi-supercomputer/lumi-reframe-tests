import os
import reframe as rfm
import reframe.utility.sanity as sn
from hpctestlib.sciapps.gromacs.benchmarks import gromacs_check

@rfm.simple_test
class lumi_gromacs_infinityhub_container(gromacs_check):
    descr = """This is to test single node, threadMPI version of GROMACS. 
    This check runs UNOFFICIAL code version from AMD's container 
    available at: https://www.amd.com/en/technologies/infinity-hub/gromacs""" 
    maintainers = ['mszpindler']
    use_multithreading = False
    num_nodes = 1
    exclusive_access = True
    time_limit = '15m'

    # This check is only for gpu
    nb_impl = parameter(['gpu'])

    benchmark_info = parameter([
        ('benchMEM', -1.08735e+06, 0.001), 			# atoms 81,743
    ], fmt=lambda x: x[0], loggable=True)
    
    executable = 'gmx mdrun'

    allref = {
        'lumi:gpu': {
            'benchMEM': (210.0, -0.075, 0.075, 'ns/day'),
        },
    }

    @deferrable
    def energy_benchpep(self):
        return sn.extractsingle(r'\s+Total Energy\s+Conserved En\.\s+Temperature\s+Pressure \(bar\)\s+Constr\. rmsd\n'
                                r'\s+(?P<energy>\S+)\s+.*',
                                'md.log', 'energy', float, item=-1)

    @deferrable
    def energy_benchmem(self):
        return sn.extractsingle(r'\s+Kinetic En\.\s+Total Energy\s+Conserved En\.\s+Temperature\s+Pressure \(bar\)\n'
                                r'(\s+\S+)\s+(?P<energy>\S+)\s+.*',
                                'md.log', 'energy', float, item=-1)


    @run_after('init')
    def setup_run(self):
        if self.nb_impl == 'gpu':
            self.container_platform = 'Singularity'
            self.valid_systems = ['lumi:gpu']
            self.valid_prog_environs = ['builtin']
            self.num_gpus_per_node = 8
            self.num_cpus_per_gpu = 6

    @run_after('init')
    def overwrite_cmds(self):
    # Need to overwrite because of downloads being not possible on compute nodes
        if self.nb_impl == 'gpu':
            self.env_vars = {
                'SINGULARITYENV_GMX_GPU_DD_COMMS': 'true', 
                'SINGULARITYENV_GMX_GPU_PME_PP_COMMS': 'true', 
                'SINGULARITYENV_GMX_FORCE_UPDATE_DEFAULT_GPU': 'true', 
                'SINGULARITYENV_OMP_NUM_THREADS': f'{self.num_cpus_per_gpu}', 
            }
        
        self.executable_opts = [
            '-notunepme', 
            '-resethway',
            '-pin', 'on',
            '-pinstride', '1',
            '-pinoffset', '1',
            '-nb', f'{self.nb_impl}',
            '-bonded', f'{self.nb_impl}',
            '-pme', f'{self.nb_impl}',
            '-update', 'cpu',
            '-v',
            '-s', 
            'benchmark.tpr'
        ]

        if self.nb_impl == 'gpu':
            self.executable_opts += ['-npme', '1',]
            self.executable_opts += ['-ntomp', f'{self.num_cpus_per_gpu}',]
            self.executable_opts += ['-ntmpi', f'{self.num_gpus_per_node}',]
            self.executable_opts += ['-gpu_id', '4,5,2,3,6,7,0,1']

    @run_before('run')
    def set_cpu_binding(self):
        if self.nb_impl == 'gpu':
            self.job.options = [f'--cpus-per-gpu={self.num_cpus_per_gpu}']
            self.job.launcher.options = ['--cpu-bind="mask_cpu:0xfefefefefefefefe"']

    @run_before('run')
    def overwrite_prerun(self):
        self.prerun_cmds = [
            f'curl -LJO https://www.mpinat.mpg.de/{self.bench_name}',
            f'unzip {self.bench_name}.zip',
            f'ln -sf {self.bench_name}.tpr benchmark.tpr'
        ]

    @run_before('run')
    def set_runtime(self):
        exec_cmd = ' '.join([self.executable, *self.executable_opts])
        if self.nb_impl == 'gpu':
            self.container_platform.image = '/project/project_462000008/reframe_resources/gromacs-infinity-hub/gromacs_2022.3.amd1_174.sif'
            self.container_platform.command = exec_cmd

    @run_before('run')
    def set_reference(self):
        self.reference = {
            '*': {
                'perf': self.allref[self.current_partition.fullname][self.bench_name]
            }
        }

@rfm.simple_test
class lumi_gromacs_mpi(gromacs_check):
    descr = """This is to test MPI version of GROMACS.""" 
    maintainers = ['mszpindler']
    use_multithreading = False
    executable_opts += ['-dlb yes', '-ntomp 1', '-npme -1']
    valid_prog_environs = ['cpeGNU']
    valid_systems = ['lumi:standard']
    modules = ['GROMACS']
    time_limit = '15m'

    num_nodes = parameter([1, 2, 4], loggable=True)

    nb_impl = parameter(['cpu'])
    
    allref = {
        # Results collected with GROMACS 2023.2 and cpeGNU/22.12
        1: {
            'HECBioSim/Crambin': (320.0, -0.05, 0.05, 'ns/day'),
            'HECBioSim/Glutamine-Binding-Protein': (115.503, -0.05, 0.05, 'ns/day'),  
            'HECBioSim/hEGFRDimer': (18.938, -0.1, 0.1, 'ns/day'),
            'HECBioSim/hEGFRDimerSmallerPL': (33.785, -0.1, 0.1, 'ns/day'),
            'HECBioSim/hEGFRDimerPair': (7.759, -0.1, 0.1, 'ns/day'),
        },
        2: {
            'HECBioSim/Crambin': (632.0, -0.05, 0.05, 'ns/day'),
            'HECBioSim/Glutamine-Binding-Protein': (237.0, -0.05, 0.05, 'ns/day'),
            'HECBioSim/hEGFRDimer': (26.8, -0.05, 0.05, 'ns/day'),
            'HECBioSim/hEGFRDimerPair': (13.4,-0.1, 0.1, 'ns/day'),
        },
        4: {
            'HECBioSim/Crambin': (576.5, -0.05, 0.05, 'ns/day'),
            'HECBioSim/Glutamine-Binding-Protein': (375.0, -0.05, 0.05, 'ns/day'),
            'HECBioSim/hEGFRDimer': (47.5, -0.05, 0.05, 'ns/day'),
            'HECBioSim/hEGFRDimerPair': (24.5, -0.1, 0.1, 'ns/day'),
            'HECBioSim/hEGFRtetramerPair': (11.1, -0.1, 0.1, 'ns/day'),
        },
    }

    @run_before('run')
    def setup_run(self):
        try:
            found = self.allref[self.num_nodes][self.bench_name]
        except KeyError:
            self.skip(f'No reference performance results for {self.bench_name!r} with {self.num_nodes}')
        self.num_tasks_per_node = 128
        self.num_tasks = self.num_nodes * self.num_tasks_per_node
        self.reference = {
            '*': {
                'perf': self.allref[self.num_nodes][self.bench_name]
            }
        }
