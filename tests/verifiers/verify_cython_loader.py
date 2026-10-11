"""Cython loader failure/off/instance-isolation gate (no compiler or NPU needed).

Run: py -3.14 tests/verifiers/verify_cython_loader.py
Each case runs in a subprocess with a timeout. Build/import paths are mocked;
this gate never deletes or rebuilds an installed .pyd/.so.
"""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
CASES = [
    "off", "invalid", "prebuilt", "auto-failure", "force-failure",
    "auto-systemexit", "force-systemexit", "force-missing-setup",
    "force-postbuild-missing", "direct-build-cache", "keyboard-interrupt",
    "m2-force", "m3-force", "model-force", "model-isolation", "train-force", "setup-paths",
]


def fresh_loader():
    spec = importlib.util.spec_from_file_location("isolated_cykernels", ROOT / "phdnet/cykernels.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def run_case(name):
    cyk = fresh_loader()
    ext = SimpleNamespace(selftest=lambda: True)
    original_argv = sys.argv
    if name == "off":
        with patch.object(cyk, "_try_import", side_effect=AssertionError("off imported")), \
             patch.object(cyk, "build_kernels", side_effect=AssertionError("off built")):
            assert cyk.get_kernels("off") is None
            assert not cyk.Kernels().active
        return
    if name == "invalid":
        cyk._kernels = ext
        expect(ValueError, lambda: cyk.get_kernels("invalid"))
        return
    if name == "prebuilt":
        with patch.object(cyk, "_try_import", return_value=ext) as imp, \
             patch.object(cyk, "build_kernels", side_effect=AssertionError("built")):
            assert cyk.get_kernels("force") is ext
            assert cyk.get_kernels("auto") is ext
            assert imp.call_count == 1
        return
    if name.startswith("m2-") or name.startswith("m3-"):
        sys.path.insert(0, str(ROOT))
        from phdnet import sparse_pc, stdp_kernels
        mod = sparse_pc if name.startswith("m2-") else stdp_kernels
        with patch.object(mod, "_get_cyk", side_effect=RuntimeError("mock failure")):
            assert mod.cyk_init("auto") is False
            expect(RuntimeError, lambda: mod.cyk_init("force"))
            # A failed force request must not become a cached silent fallback.
            expect(RuntimeError, lambda: mod.cyk_init("force"))
        with patch.object(mod, "_get_cyk", return_value=None):
            expect(RuntimeError, lambda: mod.cyk_init("force"))
        return
    if name.startswith("model-"):
        model_case(name)
        return
    if name == "setup-paths":
        # Capture setup metadata without Cython/code generation or a compiler.
        import runpy
        setup_spec = importlib.util.spec_from_file_location("isolated_setup", ROOT / "setup_cython.py")
        setup_mod = importlib.util.module_from_spec(setup_spec)
        setup_spec.loader.exec_module(setup_mod)
        captured = []
        fake_numpy = SimpleNamespace(get_include=lambda: "mock-include")
        fake_setuptools = SimpleNamespace(Extension=lambda name, **kw: captured.append(kw) or kw,
                                          setup=lambda **kw: captured.append(kw))
        fake_cython = SimpleNamespace(cythonize=lambda extensions, **kw: extensions)
        cwd = Path.cwd()
        with patch.dict(sys.modules, {"numpy": fake_numpy, "setuptools": fake_setuptools,
                                      "Cython.Build": fake_cython}):
            setup_mod._ext_modules()
            assert Path(captured[0]["sources"][0]) == ROOT / "phdnet/_cykernels.pyx"
            assert Path(captured[0]["sources"][0]).is_absolute()
            runpy.run_path(str(ROOT / "setup_cython.py"), run_name="__main__")
        meta = captured[-1]
        assert meta["package_dir"]["phdnet"] == str(ROOT / "phdnet")
        assert Path.cwd() == cwd
        return
    if name == "train-force":
        # Execute only the actual startup try/except, not the training entry.
        tree = ast.parse((ROOT / "train/train.py").read_text(encoding="utf-8"))
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.Try)
                    and any(isinstance(x, ast.Attribute) and x.attr == "get_kernels"
                            for x in ast.walk(n)))
        code = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                       "train-cython-startup", "exec")
        sys.path.insert(0, str(ROOT))
        from phdnet import cykernels
        with patch.object(cykernels, "get_kernels", side_effect=RuntimeError("mock failure")):
            expect(RuntimeError, lambda: exec(code, {"cfg": SimpleNamespace(cython_kernels="force")}))
            exec(code, {"cfg": SimpleNamespace(cython_kernels="auto")})
        return
    if name == "force-missing-setup":
        with patch.object(cyk, "_try_import", return_value=None), \
             patch.object(cyk.os.path, "isfile", return_value=False):
            expect(RuntimeError, lambda: cyk.get_kernels("force"))
        assert sys.argv is original_argv
        return
    if name == "direct-build-cache":
        with patch.object(cyk, "_try_import", return_value=ext), \
             patch("runpy.run_path", return_value={}) as build:
            assert cyk.build_kernels() is ext
            assert cyk.get_kernels("auto") is ext
            assert build.call_count == 1
        assert sys.argv is original_argv
        return
    if name == "force-postbuild-missing":
        with patch.object(cyk, "_try_import", return_value=None), \
             patch("runpy.run_path", return_value={}):
            expect(RuntimeError, lambda: cyk.get_kernels("force"))
        assert sys.argv is original_argv
        return
    if name == "keyboard-interrupt":
        with patch.object(cyk, "_try_import", return_value=None), \
             patch("runpy.run_path", side_effect=KeyboardInterrupt):
            expect(KeyboardInterrupt, lambda: cyk.get_kernels("auto"))
        assert sys.argv is original_argv
        return
    error = SystemExit(2) if "systemexit" in name else RuntimeError("compiler missing")
    mode = "force" if name.startswith("force-") else "auto"
    with patch.object(cyk, "_try_import", return_value=None), \
         patch("runpy.run_path", side_effect=error) as build:
        if mode == "force":
            expect(RuntimeError, lambda: cyk.get_kernels(mode))
        else:
            assert cyk.get_kernels(mode) is None
            assert cyk.get_kernels(mode) is None
            assert build.call_count == 1
        assert "build" in cyk._status
    assert sys.argv is original_argv


def model_case(name):
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from phdnet import cykernels, sparse_pc as spc, stdp_kernels as sk
    from phdnet.config import PHDNetConfig
    from phdnet.model import PHDNet

    def cfg(mode):
        c = PHDNetConfig()
        c.n_input, c.n_sdr, c.n_mid, c.n_top = 8, 8, 6, 4
        c.k_sparse, c.conn_k, c.m_lateral = 2, 2, 2
        c.vocab_size, c.readout_dtype = 9, "fp32"
        c.backend, c.accel_device = "numpy", "cpu"
        c.cython_kernels, c.step_profiling = mode, False
        return c

    if name == "model-force":
        with patch.object(cykernels, "get_kernels", side_effect=RuntimeError("mock failure")):
            expect(RuntimeError, lambda: PHDNet(cfg("force")))
        return

    calls = {"mv": 0, "predict": 0, "delta": 0}

    def mv(ip, ix, vl, x, out):
        calls["mv"] += 1
        out[:] = 7

    def predict(W, idx, pre, out):
        calls["predict"] += 1
        out[:] = 0.25

    def delta(*args):
        calls["delta"] += 1

    ext = SimpleNamespace(csr_matvec=mv, predict_edges=predict, stdp_delta=delta)
    with patch.object(cykernels, "get_kernels", return_value=ext) as load, \
         patch("phdnet.plasticity.NUMBA_OK", True):
        auto = PHDNet(cfg("auto"))
        off = PHDNet(cfg("off"))
        assert load.call_count == 1  # off never touches loader, even after auto.
        assert auto.pc._cyk is ext and auto.stdp._cyk is ext
        assert off.pc._cyk is None and off.stdp._cyk is None
        # Global compatibility switches cannot affect existing instances.
        spc._CYK, sk._CYK = ext, ext
        x = np.ones(8, np.float32)
        pre = np.ones(4, np.float32)
        auto.pc._mv(auto.pc.up0, x)
        auto.stdp.predict(pre)
        auto.stdp.step(pre, pre)
        assert calls == {"mv": 1, "predict": 1, "delta": 1}
        before = calls.copy()
        off.pc._mv(off.pc.up0, x)
        off.stdp.predict(pre)
        off.stdp.step(pre, pre)
        assert calls == before
        spc.cyk_init("off")
        sk.cyk_init("off")
        auto.pc._mv(auto.pc.up0, x)
        auto.stdp.predict(pre)
        auto.stdp.step(pre, pre)
        assert calls == {"mv": 2, "predict": 2, "delta": 2}
        auto2 = PHDNet(cfg("auto"))
        assert auto2.pc._cyk is ext and off.pc._cyk is None
        off.pc._mv(off.pc.up0, x)
        off.stdp.predict(pre)
        off.stdp.step(pre, pre)
        assert calls == {"mv": 2, "predict": 2, "delta": 2}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--case":
        run_case(sys.argv[2])
        print("PASS", sys.argv[2], flush=True)
        return 0
    failures = []
    env = dict(os.environ, PYTHONIOENCODING="utf-8", NUMBA_NUM_THREADS="2")
    for name in CASES:
        try:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--case", name],
                                    cwd=ROOT, env=env, timeout=60, capture_output=True,
                                    text=True, encoding="utf-8", errors="replace")
            if result.returncode:
                failures.append(name)
                print("FAIL", name, result.stdout, result.stderr, flush=True)
            else:
                print("PASS", name, flush=True)
        except subprocess.TimeoutExpired:
            failures.append(name)
            print("FAIL", name, "subprocess timeout (possible loader deadlock)", flush=True)
    print(f"cython loader: {len(CASES) - len(failures)}/{len(CASES)} PASS", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
