import os
import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher


def set_container_platform(test):
    test.container_platform = 'Singularity'
    test.container_platform.image = '$SIFPYTORCH'

class bert_fetch_data_tokenizer(rfm.RunOnlyRegressionTest):
    '''Fixture for fetching the the dataset and tokenizer'''
    modules = ['PyTorch/2.2.0-rocm-5.6.1-python-3.10-singularity-DSBertVenv-20240315']
    local = True
    env_vars = {'HF_HOME': './huggingface_home'}
    executable = 'bash ./conda-python-distributed.sh -u bert_squad_deepspeed_train.py --download-only'

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
        'TRANSFORMERS_OFFLINE': '1',
        'HF_DATASETS_OFFLINE': '1'
    }

    reference = {
        'lumi:gpu': {
            'samples_per_sec': (5579, -0.05, None, 'samples/sec')}
    }

    @sanity_function
    def assert_finished(self):
        return sn.all([
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
    def set_cpu_binding(self):
        self.job.launcher.options = ['--cpu-bind="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"']

    @run_before('run')
    def set_container_variables(self):
        set_container_platform(self)
        self.container_platform.command = 'bash ./conda-python-distributed.sh -u bert_squad_deepspeed_train.py --deepspeed_config ds_config.json --num-epochs 5 --deepspeed'

    @run_before('run')
    def prepare_job(self):
        bert_cache_dir = os.path.join(self.bert_cache.stagedir, 'cache')
        self.executable_opts.append(f'--bert-cache-dir {bert_cache_dir}')
        self.env_vars.update({
            'HF_HOME': os.path.join(self.bert_cache.stagedir,
                                    'huggingface_home')
        })

