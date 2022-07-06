import reframe as rfm
from hpctestlib.sciapps.gromacs.benchmarks import gromacs_check


@rfm.simple_test
class lumi_gromacs_check(gromacs_check):
    maintainers = ['mszpindler']
    use_multithreading = False
    executable_opts += ['-dlb yes', '-ntomp 1', '-npme -1']
    pe_release = '21.12'
    gromacs_version = '2021.5'
    valid_prog_environs = ['cpeGNU']

    num_nodes = parameter([1, 2, 4], loggable=True)
    allref = {
        1: {
            'gfx908a': { # Copied from CSCS check for sm_60
                'HECBioSim/Crambin': (195.0, None, None, 'ns/day'),
                'HECBioSim/Glutamine-Binding-Protein': (78.0, None, None, 'ns/day'),   
                'HECBioSim/hEGFRDimer': (8.5, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimerSmallerPL': (9.2, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimerPair': (3.0, None, None, 'ns/day'),
            },
            'zen3': { # Copied from CSCS check for zen2 
                'HECBioSim/Crambin': (320.0, None, None, 'ns/day'),
                'HECBioSim/Glutamine-Binding-Protein': (120.0, None, None, 'ns/day'),  
                'HECBioSim/hEGFRDimer': (16.0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimerSmallerPL': (31.0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimerPair': (7.0, None, None, 'ns/day'),
            },
        },
        2: {
            'gfx908a': { # Copied from CSCS check for sm_60
                'HECBioSim/Crambin': (202.0, None, None, 'ns/day'),
                'HECBioSim/Glutamine-Binding-Protein': (111.0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimer': (15.0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimerSmallerPL': (18.0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimerPair': (6.0, None, None, 'ns/day'),
            },
            'zen3': { # Copied from CSCS check for zen2 
                'HECBioSim/Crambin': (355.0, None, None, 'ns/day'),
                'HECBioSim/Glutamine-Binding-Protein': (210.0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimer': (31.0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimerSmallerPL': (53.0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimerPair': (13.0, None, None, 'ns/day'),
            },
        },
        4: {
            'gfx908a': { # Copied from CSCS check for sm_60
                'HECBioSim/Crambin': (200.0, None, None, 'ns/day'),
                'HECBioSim/Glutamine-Binding-Protein': (133.0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimer': (22.0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimerSmallerPL': (28.0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimerPair': (10.0, None, None, 'ns/day'),
                'HECBioSim/hEGFRtetramerPair': (5.0, None, None, 'ns/day'),
            },
            'zen3': { # Copied from CSCS check for zen2 
                'HECBioSim/Crambin': (340.0, None, None, 'ns/day'),
                'HECBioSim/Glutamine-Binding-Protein': (230.0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimer': (56.0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimerSmallerPL': (80.0, None, None, 'ns/day'),
                'HECBioSim/hEGFRDimerPair': (25.0, None, None, 'ns/day'),
                'HECBioSim/hEGFRtetramerPair': (11.0, None, None, 'ns/day'),
            },
        },
    }

    @run_after('init')
    def setup_modules(self):
        if self.current_system.name in ('lumi'):
            if self.nb_impl == 'cpu':
                self.modules = ['GROMACS-{gromacs_version}-cpeGNU-{pe_release}-CPU']
                self.valid_prog_environs = ['cpeGNU']
            # Add module information for GPU enabled version    
            #elif self.nb_impl == 'gpu':
            #    self.modules = ['GROMACS-{gromacs_version}-cpeGNU-{pe_release}-GPU']
            #    self.valid_prog_environs = ['cpeGNU']

    @run_after('init')
    def setup_filtering_criteria(self):
        # Update test's description
        self.descr += f' ({self.num_nodes} node(s))'

        # Setup system filtering
        valid_systems = {
            'cpu': {
                1: ['lumi:cpu'],
                2: ['lumi:cpu'],
                4: ['lumi:cpu'],
            },
            'gpu': {
                1: ['lumi:gpu'],
                2: ['lumi:gpu'],
                4: ['lumi:gpu'],
            }
        }
        try:
            self.valid_systems = valid_systems[self.nb_impl][self.num_nodes]
        except KeyError:
            self.valid_systems = []

        # Setup prog env. filtering
        #if self.current_system.name in ('eiger', 'pilatus'):
        #    self.valid_prog_environs = ['cpeGNU']
        #elif self.current_system.name in ('ares'):
        #    self.valid_prog_environs = ['gnu']

        #if self.num_nodes in (6, 16):
        #    self.tags |= {'production'}
        #    if (self.nb_impl == 'gpu' and
        #        self.bench_name == 'HECBioSim/hEGFRDimerSmallerPL'):
        #        self.tags |= {'maintenance'}

    @run_before('run')
    def setup_run(self):
        self.skip_if_no_procinfo()

        # Setup GPU run
        if self.nb_impl == 'gpu':
            self.num_gpus_per_node = 1

        proc = self.current_partition.processor

        # Choose arch; we set explicitly the GPU arch, since there is no
        # auto-detection
        arch = proc.arch
        if self.current_partition.fullname in ('lumi:gpu'):
            arch = 'gfx908a'

        try:
            found = self.allref[self.num_nodes][arch][self.bench_name]
        except KeyError:
            self.skip(f'Configuration with {self.num_nodes} node(s) of '
                      f'{self.bench_name!r} is not supported on {arch!r}')

        # Setup performance references
        self.reference = {
            '*': {
                'perf': self.allref[self.num_nodes][arch][self.bench_name]
            }
        }

        # Setup parallel run
        self.num_tasks_per_node = proc.num_cores
        self.num_tasks = self.num_nodes * self.num_tasks_per_node
