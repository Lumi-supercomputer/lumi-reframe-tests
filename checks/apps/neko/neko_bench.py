# adapted from Dardel (author: javier Aguilar Fruto)

import reframe as rfm
import reframe.utility.sanity as sn
import os
import csv
import string

class NekoError(Exception):
    pass

class MakeNeko(rfm.core.buildsystems.BuildSystem):
    srcfile = variable(str, type(None), value=None)

    def __init__(self):
        self.makeneko = 'makeneko'

    def emit_build_commands(self, environ):
        if not self.srcfile:
            raise NekoError('Source file required')

        return [f'{self.makeneko} "{self.srcfile}"']

class lumi_make_neko(MakeNeko, rfm.CompileOnlyRegressionTest):
    case = variable(str)
    modules = ['Neko']

    @run_after('setup')
    def set_build(self):
        self.build_system = MakeNeko()
        self.sourcepath = f'{self.case}.f90'

    @sanity_function
    def check_build(self):
        return sn.assert_found('Building user NEKO ... done!', self.stdout)

class NekoTGVBase(rfm.RunOnlyRegressionTest):
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['cpeCray']
    exclusive_access = True
    time_limit = '15m'

    # latest check with version 0.9.1-cpeCray-24.03-rocm
    modules = ['Neko']
    case = 'tgv'
    container_platform = 'Singularity'

    makeneko = fixture(lumi_make_neko, scope='environment', variables={'case': case})

    num_nodes = parameter([1,2,4,8,16])
    num_gpus_per_node = 8

    size = parameter([32768, 262144], loggable=True)
    
    mesh_file = variable(str, value='')

    # Set dofs to enable workrate perf var
    dofs = variable(int, value=0)
    first_workrate_timestep = variable(int, value=0)

    @run_before('compile')
    def set_mesh_file(self):
        self.mesh_file = f'{str(self.size)}.nmsh'
        mesh_file_path = os.path.join(self.current_system.resourcesdir, 
                                       'datasets',
                                       'neko', 
                                       'examples',
                                       'tgv',
                                       self.mesh_file)

        self.prerun_cmds += [
            f'ln -s {mesh_file_path} .'
        ]
   
    @run_after('init')
    def set_environment(self):
        self.env_vars = {
            'MPICH_GPU_SUPPORT_ENABLED': '1',
        }

    @run_before('run')
    def set_dofs(self):
        self.dofs = 8**3 * self.size

    @run_before('run')
    def add_executable(self):
        self.executable = os.path.join(self.makeneko.stagedir,
                                       'neko')
        case_file = os.path.join(self.stagedir, 
                                 str(self.size),
                                 f'{self.case}.case')
        self.executable_opts.append(case_file)

    @run_before('run')
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
    def set_num_tasks(self):
        self.num_tasks_per_node = self.num_gpus_per_node
        self.num_tasks = self.num_nodes*self.num_tasks_per_node

    @run_before('run')
    def set_cpu_binding(self):
        cpu_bind_mask = '0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000'
        self.job.launcher.options = [f'--cpu-bind=mask_cpu:{cpu_bind_mask}']

    
    @run_before('run')
    def ccpe_image(self):
        self.container_platform.image = '$SIFCCPE'
        self.executable = os.path.join(self.makeneko.stagedir,
                                       'neko')
        case_file = os.path.join(self.stagedir, 
                                 str(self.size),
                                 f'{self.case}.case')
        self.container_platform.command = './select_gpu ' + self.executable + ' ' + case_file

    @run_before('run')
    def ccpe_adapt_srun(self):
        self.job.launcher.modifier = 'SINGULARITYENV_PATH=$PATH SINGULARITYENV_LD_LIBRARY_PATH=$LD_LIBRARY_PATH'

    @sanity_function
    def normal_end(self):
        return sn.assert_found('Normal end.', self.stdout)

    @run_before('performance')
    def set_time_perf(self):
        timesteps = sn.extractall(r'Total elapsed time \(s\):\s\s\s+(\S+)', self.stdout, 1, float)

        pf = sn.make_performance_function(lambda: timesteps[-1], 's')
        self.perf_variables['total_runtime'] = pf

        if self.dofs != 0:
            pes = self.num_tasks

            def workrate():
                end = sn.count(timesteps) - 1
                time = timesteps[end] - timesteps[self.first_workrate_timestep]
                iters = end - self.first_workrate_timestep
                return 1e-3 * self.dofs * iters / time / pes

            pf = sn.make_performance_function(workrate, 'Mdofs/s/pe')
            self.perf_variables['workrate'] = pf

@rfm.simple_test
class lumi_neko_bench(NekoTGVBase):
    first_workrate_timestep = 1200

    allref = {
        32768: {
            1: {
                'lumi:gpu': {
                    'total_runtime': (131, -0.50, 0.05, 's'),
                    'workrate': (65000, -0.05, 0.05, 'Mdofs/s/pe'),
                }
            },
            2: {
                'lumi:gpu': {
                    'total_runtime': (80, -0.50, 0.05, 's'),
                    'workrate': (53000, -0.05, 0.05, 'Mdofs/s/pe'),
                }
            },
            4: {
                'lumi:gpu': {
                    'total_runtime': (54, -0.50, 0.05, 's'),
                    'workrate': (39000, -0.05, 0.05, 'Mdofs/s/pe'),
                }
            }
        },
       262144: {
            8: {
                'lumi:gpu': {
                    'total_runtime': (563, -0.50, 0.05, 's'),
                    'workrate': (62200, -0.05, 0.05, 'Mdofs/s/pe'),
                }
            },
            16: {
                'lumi:gpu': {
                    'total_runtime': (432, -0.50, 0.05, 's'),
                    'workrate': (48488, -0.05, 0.05, 'Mdofs/s/pe'),
                }
            },
       } 
    }

    @run_before('run')
    def select_tests(self):
        try:
             found = self.allref[self.size][self.num_nodes]
        except KeyError:
            self.skip(f'Test for check of size {self.size} with {self.num_nodes} nodes skipped')

    @run_before('performance')
    def set_reference(self):
        self.reference = self.allref[self.size][self.num_nodes]
