import reframe as rfm
import reframe.utility.sanity as sn

@rfm.simple_test
class lumi_train_gpt2(rfm.RegressionTest):
    descr = '''This is based on the llm.c from github.com/karpathy/llm.c and https://github.com/anthonix/llm.c
        and training description from https://github.com/karpathy/llm.c/discussions/481'''
    build_system = 'SingleSource'
    valid_systems = ['lumi:gpu']
    valid_prog_environs = ['cpeAMD']
    modules =['rocm/6.2.2']
    num_tasks = 8
    num_tasks_per_node = 8
    num_gpus_per_node = 8
    exclusive_access = True

    max_steps = variable(int, value=300)

    @run_before('compile')
    def set_compiler_flags(self):
    # hipcc -O2 -I. -fno-strict-aliasing --offload-arch=gfx90a -DMULTI_GPU -DUSE_MPI -I$MPICH_DIR/include -DENABLE_BF16 train_gpt2_amd.hip -lamdhip64 -lhipblaslt -L$MPICH_DIR/lib -lmpi -lrccl -o train_gpt2_amd
        self.sourcesdir = 'src/hip'
        self.build_system.srcfile = 'train_gpt2_amd.cu'
        self.build_system.executable = 'train_gpt2_amd'
        self.build_system.nvcc = 'hipcc'
        self.build_system.include_path = ['.', '$MPICH_DIR/include']
        self.build_system.cppflags = ['-DMULTI_GPU', '-DUSE_MPI', '-DENABLE_BF16']
        self.build_system.cxxflags = ['-O2', '-fno-strict-aliasing', '--offload-arch=gfx90a']
        self.build_system.ldflags = ['-lamdhip64', '-lhipblaslt', '-L$MPICH_DIR/lib', '-lmpi', '-lrccl']

    @run_before('run')
    def link_datasets(self):
        self.prerun_cmds = [
            f'ln -s {self.current_system.resourcesdir}/datasets/gpt2/gpt2_tokenizer.bin .',
            f'ln -s {self.current_system.resourcesdir}/datasets/hellaswag/hellaswag_val.bin .',
        ]

    @run_before('run')
    def set_executable(self):
        self.executable_opts = [
            f'-i "{self.current_system.resourcesdir}/datasets/fineweb10B/fineweb_train_*.bin"',
            f'-j "{self.current_system.resourcesdir}/datasets/fineweb10B/fineweb_val_*.bin"',
            f'-e {self.current_system.resourcesdir}/datasets/gpt2/gpt2_124M_bf16.bin',
            '-o log_gpt2_124M',
            '-v 250 -s 20000 -g 144 -h 1 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 1 -e "d12"',
            f'-x {self.max_steps}'
        ]
        self.executable = self.build_system.executable

    @performance_function('ms')
    def average_iteration_time(self):
        return sn.extractsingle(rf'total average iteration time:\s+(\S+)\sms', self.stdout, 1, float)

    @performance_function('tok/s')
    # reads tokens per second achieved at last iterarion (max_steps)
    def tokens_per_sec_at_last(self):
        return sn.extractsingle(rf'step\s+{self.max_steps}\/{self.max_steps}\s\|\s\S+\s\S+\s\(\S+\)\|{2}\s\S+\s\S+\s\|{2}\s\S+\s\S+\s\S+\s|\s(\S+)\stok\/s', self.stdout, 1, int)

    @sanity_function
    # checks if there are multiple accurences of `val loss` in the output and if it decreases
    def compare_loss(self):
        losses = sn.extractall(r'val loss\s(\S+)', self.stdout, 1, float)
        return sn.all([
            sn.assert_gt(sn.count(losses), 1),
            sn.assert_lt(losses[sn.count(losses)-1], losses[0])
        ])
