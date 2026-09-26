// GitCode 流水线（Jenkins 声明式语法）——与 GitHub Actions 的 ci.yml 同一套测试。
// GitCode 控制台启用方式：仓库 → 持续集成 → 选择「Jenkins」并指向本文件；
// 若仓库启用的是 GitCode 原生流水线，可用同一命令序列配置（见 ci/run_tests.py）。
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
                '''
            }
        }
        stage('回归测试 + 逐位对拍 + 精度验证') {
            steps {
                sh 'python3 ci/run_tests.py'
            }
        }
        stage('硬件后端适配验证（CPU 参考）') {
            steps {
                sh '''
                python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu || true
                python3 tools/bench_accel.py --steps 20
                python3 -c "from phdnet.torch_backend import selftest_torch; assert selftest_torch('cpu'); print('selftest PASS')"
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
