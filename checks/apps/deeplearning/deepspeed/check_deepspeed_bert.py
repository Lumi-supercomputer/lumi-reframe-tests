import os
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher


def set_container_platform(test):
    test.container_platform = 'Singularity'
    test.container_platform = 'Singularity'
    test.container_platform.image = '$SIFPYTORCH'

class bert_fetch_data_tokenizer(rfm.RunOnlyRegressionTest):
    '''Fixture for fetching the the dataset and tokenizer'''
    modules = ['PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-DSBertVenv-20240315']
    local = True
    env_vars = {'HF_HOME': './huggingface_home'}
    executable = 'conda-python-simple -u bert_squad_deepspeed_train.py --download-only'

    @sanity_function
    def validate_download(self):
        return sn.assert_eq(self.job.exitcode, 0)

    @run_before('run')
    def set_container_variables(self):
        set_container_platform(self)
        self.container_platform.command = self.executable


class deepspeed_bert_qa_train_base(rfm.RunOnlyRegressionTest):
    descr = ('Check the training throughput of a BERT with Squad for the QA '
             'task with DeepSpeed')
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['builtin']
    sourcesdir = 'src'
    exclusive_access = True
    num_tasks = 16
    num_tasks_per_node = 8
    num_gpus_per_node = 8
    env_vars = {
        'NCCL_DEBUG': 'INFO',
        'TORCH_EXTENSIONS_DIR': './torch_extensions',
        'TRANSFORMERS_OFFLINE': '1',
        'HF_DATASETS_OFFLINE': '1'
    }
    executable = 'conda-python-distributed'
    executable_opts = ['-u bert_squad_deepspeed_train.py', '--deepspeed_config ds_config.json',
                       '--num-epochs 5', '--gpu', '--modelpath model']
    reference = {
        'lumi:gpu': {
            'samples_per_sec': (5579, -0.05, None, 'samples/sec')}
    }

    @sanity_function
    def assert_world_rank(self):
        world_ranks = sn.extractall(r'world_rank=(?P<world_rank>\S+),',
                                    self.stdout, 'world_rank', float)
        return sn.all([
            sn.assert_eq(sorted(world_ranks), list(range(self.num_tasks))),
            sn.assert_found(r'Finished Training', self.stdout)
        ])

    @performance_function('samples/sec')
    def samples_per_sec(self):
        return sn.avg(sn.extractall(
            r'RunningAvgSamplesPerSec=(?P<samples_per_sec>\S+),',
            self.stdout, 'samples_per_sec', float
        ))


@rfm.simple_test
class deepspeed_bert_qa_train(deepspeed_bert_qa_train_base):
    # Besides `deepspeed`, the packages`transformers`, `datasets` and `rich`
    # are needed to run this test
    modules = ['PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-DSBertVenv-20240315']
    bert_cache = fixture(bert_fetch_data_tokenizer, scope='session')

    tags = {'python', 'contrib'}

    @run_before('run')
    def set_container_variables(self):
        set_container_platform(self)
        self.container_platform.command = 'conda-python-distributed -u bert_squad_deepspeed_train.py --deepspeed_config ds_config.json --num-epochs 5'

    @run_before('run')
    def prepare_job(self):
        bert_cache_dir = os.path.join(self.bert_cache.stagedir, 'cache')
        self.executable_opts.append(f'--bert-cache-dir {bert_cache_dir}')
        self.env_vars.update({
            'HF_HOME': os.path.join(self.bert_cache.stagedir,
                                    'huggingface_home')
        })


#@rfm.simple_test
#class deepspeed_bert_qa_train_singularity(deepspeed_bert_qa_train_base):
    # Using a container from https://hub.docker.com/r/rocm/deepspeed
    # The container used here doesn't include all the packages needed to run
    # this test:
    # MPICC=mpicc pip install --user mpi4py
    # pip install --user datasets transformers tokenizers rich

#    bert_cache = fixture(bert_fetch_data_tokenizer_singularity,
#                         scope='session')

#    tags = {'singularity', 'contrib/22.06', 'contrib/22.08'}

#    @run_before('run')
#    def set_container_variables(self):
#        set_container_platform(self)
#        # FIXME: copy of `comm.py` that uses `hostname -i`
#        # instead of `hostname -I`
#        comm_py = os.path.join(
#            self.current_system.resourcesdir,
#            'deepspeed', 'comm.py'
#        )
#        # FIXME: The current tag misses the `quantizer_hip.h` ops file
#        quantizer_hip = os.path.join(
#            self.current_system.resourcesdir,
#            'deepspeed', 'quantizer_hip.h'
#        )
#        deepspeed_dir = '/opt/conda/lib/python3.7/site-packages/deepspeed'
#        bert_cache_dir = '/bert_cache_dir'
#        self.container_platform.mount_points = [
#            ('$SCRATCH', '$SCRATCH'),
#            (self.bert_cache.stagedir, bert_cache_dir),
#            # FIXME: Mounting missing deepspeed files or files needing changes
#            (comm_py, os.path.join(deepspeed_dir, 'comm/comm.py')),
#            (quantizer_hip, os.path.join(deepspeed_dir, 'ops/csrc/includes/quantizer_hip.h'))  # noqa E501
#        ]
#        self.container_platform.command = ' '.join([
#            self.executable,
#            *self.executable_opts,
#            '--bert-cache-dir /bert_cache_dir/cache'
#        ])
#        self.env_vars.update({
#            'HF_HOME': os.path.join(bert_cache_dir, 'huggingface_home')
#        })
#
#    @run_before('run')
#    def set_launcher(self):
#        # The job launcher has to be changed to `mpirun` since the software
#        # in the container is based on OpenMPI and it would fail with `srun`
#        self.job.launcher = getlauncher('mpirun')()
