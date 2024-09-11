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
    rocm_version = parameter(['6.0.3-dockerhash-1ada3e019646', '6.2.0-dockerhash-c8e37dff2e91', '6.1.3-dockerhash-4a063f050ed7'])

    @run_before('run')
    def set_container_variables(self):
        self.container_platform.image = f'/project/project_462000008/containers/lumi-rocm-rocm-{self.rocm_version}.sif'
        self.container_platform.command = 'rocminfo'
