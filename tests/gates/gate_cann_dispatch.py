"""Dispatch diagnostics: unknown JIT is not success; explicit blocking zero is allowed.

Uses mocked torch_npu status, not NPU hardware or performance evidence.
"""
from __future__ import annotations
import os
from pathlib import Path
import sys
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from phdnet.backends.cann_env import verify_dispatch


def probe(jit, blocking=None, rt_blocking=None, queue='2'):
    torch = types.ModuleType('torch')
    torch_npu = types.ModuleType('torch_npu')
    npu = types.ModuleType('torch_npu.npu')
    def get_jit():
        if isinstance(jit, Exception):
            raise jit
        return jit
    npu.npuConfig = types.SimpleNamespace(is_jit_compile_false=get_jit)
    env = {'TASK_QUEUE_ENABLE': queue, 'COMBINED_ENABLE': '1'}
    if blocking is not None:
        env['ASCEND_LAUNCH_BLOCKING'] = blocking
    if rt_blocking is not None:
        env['ASCEND_RT_LAUNCH_BLOCKING'] = rt_blocking
    with patch.dict(sys.modules, {'torch': torch, 'torch_npu': torch_npu,
                                 'torch_npu.npu': npu}), patch.dict(os.environ, env, clear=True):
        before = dict(os.environ)
        result = verify_dispatch(report=False)
        assert dict(os.environ) == before, 'diagnostic changed the environment'
        return result


def main():
    cases = [
        ('binary/unset', True, None, None, '2', True),
        ('explicit zero', True, '0', None, '2', True),
        ('both zeros', True, '0', '0', '2', True),
        ('JIT enabled', False, None, None, '2', False),
        ('JIT unavailable', RuntimeError('unavailable'), None, None, '2', None),
        ('JIT unknown', None, None, None, '2', None),
        ('blocking one', True, '1', None, '2', False),
        ('runtime zero cannot mask blocking one', True, '1', '0', '2', False),
        ('runtime one', True, '0', '1', '2', False),
        ('unknown blocking value', True, 'invalid', None, '2', None),
        ('known failure plus unknown JIT', None, None, None, '0', False),
    ]
    for label, jit, blocking, rt, queue, expected in cases:
        actual = probe(jit, blocking, rt, queue)['effective']
        assert actual is expected, (label, actual, expected)
        print('PASS', label, 'effective=', actual)
    print(f'cann dispatch gate: {len(cases)}/{len(cases)} passed (mocked status only)')


if __name__ == '__main__':
    main()
