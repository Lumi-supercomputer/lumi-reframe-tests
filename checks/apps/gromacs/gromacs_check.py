import os
import reframe as rfm
import reframe.utility.sanity as sn
from hpctestlib.sciapps.gromacs.benchmarks import gromacs_check

@rfm.simple_test
class lumi_gromacs_hecbiosim(gromacs_check):
    descr = """This is to test MPI version of GROMACS.""" 

    valid_systems = ['lumi:cpu']
    valid_prog_environs = ['cpeGNU']
    module_ver = parameter([
        '2023.3-cpeGNU-23.09-CPU',
    ], loggable=True)
    maintainers = ['mszpindler']
    use_multithreading = False
    executable_opts += ['-dlb yes', '-ntomp 1', '-npme -1']
    time_limit = '15m'

    num_nodes = parameter([1, 2, 4], loggable=True)

    nb_impl = parameter(['cpu'])
    tags = {'benchmark', 'contrib', 'cpu'}
    keep_files = ['md.log']
    
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

    @run_after('init')
    def apply_module_ver(self):
        module = f'GROMACS/{self.module_ver}'
        self.modules = [module]

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
