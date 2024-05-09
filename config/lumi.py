project = 'project_462000008'

site_configuration = {
    'systems': [
        {
            'name': 'lumi',
            'descr': 'LUMI Cray EX Supercomputer',
            'hostnames': ['ln\d+-nmn', 'uan\d+-nmn.local', '\S+'],
            'modules_system': 'lmod',
            'modules': ['LUMI'],
            'resourcesdir': '/projappl/%s/reframe_resources/' % project,
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'time_limit': '10m',
                    'environs': [
                        'builtin',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'cpeCray',
                        'cpeGNU',
                    ],
                    'descr': 'Login nodes',
                    'max_jobs': 4,
                    'modules': ['partition/L'],
                    'launcher': 'local'
                },
                {
                    'name': 'small',
                    'descr': 'Multicore nodes (AMD EPYC 7763, 256|512|1024GB/cn)',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'container_platforms': [
                        {
                            'type': 'Singularity',
                            'modules': []
                        }
                    ],
                    'environs': [
                        'builtin',
                        'PrgEnv-aocc',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'cpeAOCC',
                        'cpeCray',
                        'cpeGNU',
                    ],
                    'max_jobs': 100,
                    'modules': ['partition/C'],
                    'access': ['--partition small',
                               f'--account={project}'],
                    'resources': [
                        {
                            'name': 'memory',
                            'options': ['--mem={mem_per_node}']
                        },
                    ],
                    'launcher': 'srun'
                    },
                    {
                    'name': 'standard',
                    'descr': 'Multicore nodes (AMD EPYC 7763, 256GB/cn)',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'container_platforms': [
                        {
                            'type': 'Singularity',
                            'modules': []
                        }
                    ],
                    'environs': [
                        'builtin',
                        'PrgEnv-aocc',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'cpeAOCC',
                        'cpeCray',
                        'cpeGNU',
                    ],
                    'max_jobs': 100,
                    'modules': ['partition/C'],
                    'access': ['--partition standard',
                               f'--account={project}'],
                    'resources': [
                        {
                            'name': 'memory',
                            'options': ['--mem={mem_per_node}']
                        },
                    ],
                    'launcher': 'srun'
                },
                {
                    'name': 'gpu',
                    'descr': 'Multicore nodes (AMD EPYC 7A53 64-Core, 512|GB/cn), GPU (AMD Instinct MI250X 8/cn)',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'container_platforms': [
                        {
                            'type': 'Singularity',
                            'modules': []
                        }
                    ],
                    'environs': [
                        'builtin',
                        'PrgEnv-amd',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'cpeAMD',
                        'cpeCray',
                        'cpeGNU',
                        'ROCm',
                    ],
                    'max_jobs': 10,
                    'modules': ['partition/G'],
                    'access': ['--partition small-g',
                               f'--account={project}'],
                    'resources': [
                        {
                            'name': 'memory',
                            'options': ['--mem={mem_per_node}']
                        },
                        {
                            'name': '_rfm_gpu',
                            'options': ['--gpus-per-node={num_gpus_per_node}']
                        },
                    ],
                    'launcher': 'srun'
                },
            ]
        },
    ],
    'environments': [
        {
            'name': 'PrgEnv-aocc',
            'target_systems': ['lumi:small', 'lumi:standard'],
            'modules': ['PrgEnv-amd']
        },
        {
            'name': 'PrgEnv-amd',
            'target_systems': ['lumi:gpu'],
            'modules': ['PrgEnv-amd']
        },
        {
            'name': 'PrgEnv-cray',
            'target_systems': ['lumi'],
            'modules': ['PrgEnv-cray']
        },
        {
            'name': 'PrgEnv-gnu',
            'target_systems': ['lumi'],
            'modules': ['PrgEnv-gnu']
        },
        {
            'name': 'cpeAMD',
            'target_systems': ['lumi:gpu'],
            'modules': ['cpeAMD']
        },
        {
            'name': 'cpeAOCC',
            'target_systems': ['lumi:small', 'lumi:standard'],
            'modules': ['cpeAOCC']
        },
        {
            'name': 'cpeCray',
            'target_systems': ['lumi'],
            'modules': ['cpeCray']
        },
        {
            'name': 'cpeGNU',
            'target_systems': ['lumi'],
            'modules': ['cpeGNU']
        },
        {
            'name': 'ROCm',
            'cc': 'hipcc',
            'cxx': 'hipcc',
            'ftn': '',
            #'cflags': 
            #'ldflags': 
            'cppflags': ['-D__HIP_PLATFORM_AMD__'],
            'modules': ['rocm'],
            'target_systems': ['lumi:gpu']
        }
    ],
    'logging': [
        {
            'handlers': [
                {
                    'type': 'file',
                    'name': 'reframe.log',
                    'level': 'debug2',
                    'format': '[%(asctime)s] %(levelname)s: %(check_info)s: %(message)s',   # noqa: E501
                    'append': False
                },
                {
                    'type': 'stream',
                    'name': 'stdout',
                    'level': 'info',
                    'format': '%(message)s'
                },
                {
                    'type': 'file',
                    'name': 'reframe.out',
                    'level': 'info',
                    'format': '%(message)s',
                    'append': False
                }
            ],
            'handlers_perflog': [
                {
                    'type': 'filelog',
                    'prefix': '%(check_system)s/%(check_partition)s',
                    'level': 'info',
                    #check fields: name|build_locally|build_time_limit|descr|display_name|env_vars|environ|exclusive_access|executable|executable_opts|extra_resources|hashcode|job_completion_time_unix|job_exitcode|job_nodelist|jobid|keep_files|local|maintainers|max_pending_time|modules|name|num_cpus_per_task|num_gpus_per_node|num_nodes|num_tasks|num_tasks_per_core|num_tasks_per_node|num_tasks_per_socket|outputdir|partition|perf_var=perf_value perf_unit|postbuild_cmds|postrun_cmds|prebuild_cmds|prefix|prerun_cmds|readonly_files|short_name|sourcepath|sourcesdir|stagedir|strict_check|system|tags|time_limit|unique_name|use_multithreading|valid_prog_environs|valid_systems|variables
                    'format': '%(check_job_completion_time)s,%(check_short_name)s,%(check_hashcode)s,%(check_modules)s,%(check_module_ver)s,%(check_name)s,%(check_system)s,%(check_partition)s,%(check_environ)s,jobid=%(check_jobid)s,num_nodes=%(check_num_nodes)s,num_task=%(check_num_tasks)s,num_tasks_per_node=%(check_num_tasks_per_node)s,num_gpus_per_node=%(check_num_gpus_per_node)s,%(check_perfvalues)s',
                    'format_perfvars': '%(check_perf_var)s=%(check_perf_value)s %(check_perf_unit)s,',
                    'datefmt': '%FT%T%:z',
                    'append': True
                },
            ]
        }
    ],
    'modes': [
        {
            'name': 'maintenance',
            'options': [
                '--exec-policy=async',
                '--strict',
                '--output=/project/%s/$USER/regression/maintenance' % project,
                '--perflogdir=/project/%s/$USER/regression/maintenance/logs' % project,
                '--stage=/scratch/%s/regression/maintenance/stage' % project,
                '--report-file=/project/%s/$USER/regression/maintenance/reports/maint_report_{sessionid}.json' % project,
                '-Jreservation=maintenance',
                '--save-log-files',
                '--tag=maintenance',
                '--timestamp=%F_%H-%M-%S'
            ]
        },
        {
            'name': 'production',
            'options': [
                '--exec-policy=async',
                '--strict',
                '--output=/project/%s/$USER/regression/production' % project,
                '--perflogdir=/project/%s/$USER/regression/production/logs' % project,
                '--stage=/scratch/%s/regression/production/stage' % project,
                '--report-file=/project/%s/$USER/regression/production/reports/prod_report_{sessionid}.json' % project,
                '--save-log-files',
                '--tag=production',
                '--timestamp=%F_%H-%M-%S'
            ]
        }
    ],
    'general': [
        {
            'check_search_path': ['checks/'],
            'check_search_recursive': True,
            'remote_detect': False
        }
    ]
}
