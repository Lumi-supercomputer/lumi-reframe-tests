import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher

@rfm.simple_test
class MultiLaunchTest(rfm.RunOnlyRegressionTest):
    valid_systems = ['lumi:gpu', 'lumi:small']
    valid_prog_environs = ['builtin']
    executable = 'wait'
    num_tasks_per_node = 3
    num_nodes = 3
    num_tasks = num_nodes*num_tasks_per_node

    @run_before('run')
    def pre_launch(self):
        cmd = self.job.launcher.run_command(self.job)
        background_cmd = 'hostname'
        self.prerun_cmds = [
            f'{cmd} --overlap -N {self.num_nodes} -n {self.num_tasks_per_node} {background_cmd} &'
            for n in range(1, self.num_nodes+1)
        ]

    @run_before('run')
    def set_launcher(self):
        self.job.launcher = getlauncher('local')()

    @sanity_function
    def validate_test(self):
        return sn.assert_eq(
            sn.count(sn.extractall(r'^nid\d+', self.stdout)), self.num_tasks
        )
