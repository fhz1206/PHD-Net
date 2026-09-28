// GitCode Jenkins 流水线（声明式语法）——与 .gitcode/workflows/ci.yml（GitCode Action）、
// .github/workflows/ci.yml（GitHub 镜像）同一套测试。
// GitCode 控制台启用方式：仓库 → 持续集成 → 选择「Jenkins」并指向本文件。
// 验证脚本统一位于 tests/verifiers/（2026-09-28 目录治理：顶层 ci/ 已移除）。
pipeline {
    agent any

    options {
        timestamps()
        timeout(time: 40, unit: 'MINUTES')
    }

    environment {
        PYTHONUNBUFFERED = '1'
    }

    stages {
        stage('环境准备') {
            steps {
                sh '''
                python3 -m pip install --upgrade pip
                python3 -m pip install numpy numba pyarrow psutil
                # CPU 版 torch：否则两个 torch 回归检查会以「未安装」跳过
                python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
                '''
            }
        }
        stage('fast 回归（19 项，零回归门槛）') {
            steps {
                sh 'python3 tests/run_tests.py fast'
            }
        }
        stage('逐位对拍验证') {
            steps {
                sh '''
                python3 tests/verifiers/verify_seg_equiv.py
                python3 tests/verifiers/verify_vocab_parallel.py
                python3 tests/verifiers/verify_parallel_consistency.py
                '''
            }
        }
        stage('torch LM 栈等价性（容差判据）') {
            steps {
                sh 'python3 tests/verifiers/verify_torch_lm.py --device cpu'
            }
        }
        stage('硬件后端适配验证（CPU 参考）') {
            steps {
                sh '''
                python3 -c "import torch; print('torch', torch.__version__)"
                python3 tools/bench_accel.py --steps 20
                python3 -c "from phdnet.torch_backend import selftest_torch; assert selftest_torch('cpu'); print('selftest PASS')"
                python3 -c "
          import sys; sys.path.insert(0, 'tests')
          from checks_backend import t_torch_equivalence, t_torch_model_smoke
          t_torch_equivalence(); t_torch_model_smoke()
          print('torch checks PASS')"
                '''
            }
        }
    }

    post {
        failure {
            echo 'CI FAILED —— 见对应 stage 日志'
        }
        success {
            echo 'CI ALL PASS'
        }
    }
}
