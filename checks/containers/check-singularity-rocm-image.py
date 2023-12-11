import os
import reframe as rfm
import reframe.utility.sanity as sn

class singularity_rocm_image(rfm.RunOnlyRegressionTest):
    valid_systems = ['lumi:gpu']
    container_platform = 'Singularity'
    num_tasks = 1
    num_tasks_per_node = 1
    num_gpus_per_node = 8

    tags = {'production', 'singularity', 'craype'}

    @sanity_function
    def assert_gpus_found(self):
        num_gpus = sn.count(sn.findall(r'\s+Name\:\s+gfx90a', self.stdout))
        return sn.assert_eq(num_gpus, self.num_gpus_per_node)

@rfm.simple_test
class test_rocm_container(singularity_rocm_image):
    valid_prog_environs = ['builtin']
    rocm_version = parameter(['5.5.1', '5.6.1'])

    @run_before('run')
    def set_container_variables(self):
        self.container_platform.image = f'/appl/local/containers/sif-images/lumi-rocm-rocm-{self.rocm_version}.sif'
        self.container_platform.command = 'rocminfo'
