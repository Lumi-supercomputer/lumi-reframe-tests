import os
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher
from reframe.utility.osext import cray_cdt_version

class singularity_container_image(rfm.RunOnlyRegressionTest):
    descr = 'Pytorch distributed in container'
    valid_systems = ['lumi:gpu']
    container_platform = 'Singularity'
    num_tasks = 16
    num_tasks_per_node = 8
    num_gpus_per_node = 8

    @run_before('run')
    def set_launch_settings(self):
        self.env_vars = {
            'Nodes':'2',
            'SINGULARITY_BIND':'/var/spool/slurmd:/var/spool/slurmd,'
                                '/opt/cray:/opt/cray,'
                                '/usr/lib64/libcxi.so.1:/usr/lib64/libcxi.so.1,'
                                '/usr/lib64/libjansson.so.4:/usr/lib64/libjansson.so.4'
        }
        self.job.launcher.options = ['--cpu-bind=map_cpu:49,57,17,25,1,9,33,41']

@rfm.simple_test
class test_pytorch_container(singularity_container_image):
    valid_prog_environs = ['builtin']
    container = parameter(['/appl/local/containers/sif-images/lumi-pytorch-rocm-5.5.1-python-3.10-pytorch-v2.0.1.sif','/appl/local/containers/sif-images/lumi-pytorch-rocm-5.6.1-python-3.10-pytorch-v2.1.0.sif'])
    reference = {
            'lumi:gpu': {
                'training_time' : (13.00, None, 0.1, 's'),
                }
            }

    @sanity_function
    def check_last_line(self):
        return sn.assert_found(r'GPU training time', self.stderr)

    @performance_function('s')
    def training_time(self):
        return sn.extractsingle(
            r"GPU training time= \d+:\d+:(\d+\.\d+)$",
            self.stderr, 1, float
        )
    
    @run_before('run')
    def set_container_variables(self):
        self.sourcesdir = 'src/pytorch'
        self.container_platform.image = f'{self.container}'
        self.container_platform.command = 'bash run-pytorch.sh'

@rfm.simple_test
class test_tensorflow_container(singularity_container_image):
    valid_prog_environs = ['builtin']
    reference = {
            'lumi:gpu': {
                'images_sec' : (450.00, -0.1, None, 's'),
                }
            }

    @sanity_function
    def check_last_line(self):
        return sn.assert_found(r'Img/sec', self.stdout)

    @performance_function('s')
    def images_sec(self):
        return sn.extractsingle(
            r'(\d+\.\d+)\s+\+-\d+\.\d+',
            self.stdout, 1, float
        )

    @run_before('run')
    def set_container_variables(self):
        self.sourcesdir = 'src/tensorflow'
        self.container_platform.image = '/appl/local/containers/sif-images/lumi-tensorflow-rocm-5.5.1-python-3.10-tensorflow-2.11.1-horovod-0.28.1.sif'
        self.container_platform.command = 'bash run-tensorflow.sh'


@rfm.simple_test
class test_container_rccl(singularity_container_image):
    valid_prog_environs = ['builtin']
    reference = {
        'lumi:gpu': {
            'busbw': (55.76, -0.05, None, 'GB/s'),
            'algbw': (29.21, -0.05, None, 'GB/s'),
        }
    }

    @sanity_function
    def check_last_line(self):
        return sn.assert_found(r'Avg bus bandwidth', self.stdout)

    @performance_function('GB/s')
    def busbw(self):
        return sn.extractsingle(
            r'^\s+134217728.+\s+(?P<busbw>\S+)\s+\S+$',
            self.stdout, 'busbw', float
        )

    @performance_function('GB/s')
    def algbw(self):
        return sn.extractsingle(
            r'^\s+134217728.+\s+(?P<algbw>\S+)\s+\S+\s+\S+$',
            self.stdout, 'algbw', float
        )

    @run_before('run')
    def set_container_variables(self):
        self.sourcesdir = 'src/rccl'
        self.container_platform.image = '/appl/local/containers/sif-images/lumi-tensorflow-rocm-5.5.1-python-3.10-tensorflow-2.11.1-horovod-0.28.1.sif'
        self.container_platform.command = 'bash run.sh'

