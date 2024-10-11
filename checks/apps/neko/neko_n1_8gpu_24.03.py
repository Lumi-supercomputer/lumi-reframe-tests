# adapted from Dardel (author: javier Aguilar Fruto)

import reframe as rfm
import reframe.utility.sanity as sn
import os
import csv
import string

def get_gpu_device(partition):
    for device in partition.devices:
        if device.type == 'gpu':
            return device

class NekoError(Exception):
    pass


# Use this for children of NekoTestBase that don't need makeneko
class DummyBuildSystem(rfm.core.buildsystems.BuildSystem):
    def emit_build_commands(self, environ):
        return []

class MakeNeko(rfm.core.buildsystems.BuildSystem):
    srcfile = variable(str, type(None), value=None)

    def __init__(self):
        self.makeneko = 'makeneko'

    def emit_build_commands(self, environ):
        if not self.srcfile:
            raise NekoError('Source file required')

        return [f'{self.makeneko} "{self.srcfile}"']

class NekoTestBase(rfm.RegressionTest):
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['PrgEnv-cray']
    exclusive_access = True

    modules = ['Neko']
    scheme = parameter(os.getenv('NEKO_SCHEME', 'pnpn').split(','))
    case = variable(str)

    num_gpus_per_node = 8
    
    mesh_file = variable(str, value='')
    dt = variable(str, value='')
    T_end = variable(str, value='')

    abstol_vel = {'sp': '1d-5', 'dp': '1d-9'}
    abstol_prs = {'sp': '1d-5', 'dp': '1d-9'}

    # Set dofs to enable workrate perf var
    dofs = variable(int, value=0)
    first_workrate_timestep = variable(int, value=0)

    @run_before('compile')
    def copy_mesh_file(self):
        if self.mesh_file == '':
            return

        src = os.path.join(self.prefix, '..', self.mesh_file)
        dst = os.path.join(self.stagedir, self.mesh_file)
        self.postbuild_cmds += [
                f'mkdir -p {os.path.dirname(self.mesh_file)}',
                f'cp "{src}" "{dst}"'
        ]
   
    @run_after('init')
    def set_environment(self):
        self.env_vars = {
            'MPICH_GPU_SUPPORT_ENABLED': '1',
        }

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
    def make_case_file(self):
        case_file = os.path.join(self.stagedir, self.case)
        self.executable_opts.append(self.case)

    @run_before('run')
    def set_num_tasks(self):
        self.num_tasks = self.num_gpus_per_node

    @run_before('run')
    def set_cpu_binding(self):
        cpu_bind_mask = '0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000'
        self.job.launcher.options = [f'--cpu-bind=mask_cpu:{cpu_bind_mask}']

    @sanity_function
    def normal_end(self):
        return sn.assert_found('Normal end.', self.stdout)

    @run_before('performance')
    def set_time_perf(self):
        timesteps = sn.extractall(r'Elapsed time \(s\):\s+(\S+)', self.stdout, 1, float)

        pf = sn.make_performance_function(lambda: timesteps[-1], 's')
        self.perf_variables['total_runtime'] = pf

        if self.dofs != 0:
            pes = self.num_tasks

            def workrate():
                end = sn.count(timesteps) - 1
                time = timesteps[end] - timesteps[self.first_workrate_timestep]
                dofs = 8**3 * 32**3
                iters = end - self.first_workrate_timestep
                return 1e-3 * dofs * iters / time / pes

            pf = sn.make_performance_function(workrate, 'Mdofs/s/pe')
            self.perf_variables['workrate'] = pf

class GetTgvDns(rfm.RunOnlyRegressionTest):
    descr = 'Download TGV DNS data'
    executable = './get-tgv-dns.sh'
    local = True

    @run_after('run')
    def load_enstrophy(self):
        self.enstrophy = {}
        path = os.path.join(self.stagedir, 'spectral_Re1600_512.gdiag')
        with open(path, newline='') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                if row[0][0] == '#':
                    continue
                # time: value
                self.enstrophy[float(row[0])] = float(row[3])

    @sanity_function
    def check_data_count(self):
        return sn.assert_eq(sn.count(sn.defer(self.enstrophy)), 2000)

class TgvBase(NekoTestBase):
    descr = 'Run TGV and compare with DNS data'
    executable = './neko'
    case = 'tgv.case'
    tgv_dns = fixture(GetTgvDns, scope='session')

    @run_after('setup')
    def set_build(self):
        self.build_system = MakeNeko()
        self.sourcepath = 'tgv.f90'

@rfm.simple_test
class lumi_neko_tgv32(TgvBase):
    dofs = 8**3 * 32**3
    first_workrate_timestep = 1200

    @run_before('performance')
    def set_reference(self):
        self.reference = {
            'lumi:gpu': {
                'total_runtime': (131, -0.50, 0.05, 's'),
                'enstrophy_error': (9.018, -0.01, 0.01, '%'),
            }
        }

