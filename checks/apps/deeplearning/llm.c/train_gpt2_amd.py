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

    @run_before('compile')
    def set_compiler_flags(self):
    # hipcc -O2 -I. -fno-strict-aliasing --offload-arch=gfx90a -DMULTI_GPU -DUSE_MPI -I/opt/cray/pe/mpich/8.1.29/ofi/gnu/12.3/include -DENABLE_BF16 train_gpt2_amd.hip -lamdhip64 -lhipblaslt -L/opt/cray/pe/mpich/8.1.29/ofi/gnu/12.3/lib -lmpi -lrccl -o train_gpt2_amd
        self.sourcesdir = 'src/hip'
        self.build_system.srcfile = 'train_gpt2_amd.cu'
        self.build_system.executable = 'train_gpt2_amd'
        self.build_system.nvcc = 'hipcc'
        self.build_system.include_path = ['.', '$MPICH_DIR/include']
        self.build_system.cppflags = ['-DMULTI_GPU', '-DUSE_MPI', '-DENABLE_BF16']
        self.build_system.cxxflags = ['-O2', '-fno-strict-aliasing', '--offload-arch=gfx90a']
        self.build_system.ldflags = ['-lamdhip64', '-lhipblaslt', '-L$MPICH_DIR/lib', '-lmpi', '-lrccl']


    @run_before('run')
    def set_executable(self):
        self.executable_opts = [f'-i "{self.current_system.resourcesdir}/datasets/fineweb10B/fineweb_train_*.bin" -j "{self.current_system.resourcesdir}/datasets/fineweb10B/fineweb_val_*.bin" -e {self.current_system.resourcesdir}/datasets/gpt2/gpt2_124M_bf16.bin -o log_gpt2_124M -v 250 -s 20000 -g 144 -h 1 -b 64 -t 1024 -d 524288 -r 0 -z 1 -c 0.1 -l 0.0006 -q 0.0 -u 700 -n 5000 -y 1 -e "d12"']
        self.executable = self.build_system.executable
