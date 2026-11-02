"""Run each bounded campaign in a fresh process and retain every failed check."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
TESTS = ('mt25_gaussianisation', 'ou_box', 'drag_decay', 'dustywave', 'beams_free_streaming',
         'sgs_acoustic', 'turb_energy_ledger', 'particle_reference_2d', 'long_turbulence')
# long_turbulence takes ~40 min: it is rerun only with --long; otherwise its last
# full result is reported (its metrics record the source hashes it ran on).
LONG = {'long_turbulence'}
OUT = ROOT/'outputs'/'dust_validation'


def summarize(executions):
    lines = ['# Dust/SGS validation results', '',
             'Every reported check is listed below. Closure-error measurements without an',
             'analytic accuracy requirement are descriptive entries in each metrics.json.', '',
             '| Test | Runtime (s) | Checks passed / total | Process status | Figures |',
             '|---|---:|---:|---|---|']
    details, total_failed = [], 0
    for name in TESTS:
        path = OUT/name/'metrics.json'
        result = json.loads(path.read_text()) if path.exists() else None
        execution = executions.get(name, {})
        status = execution.get('status', 'existing results')
        checks = list(result.get('checks', [])) if result else []
        if status not in ('completed', 'existing results'):
            checks.append(dict(name='process completed within 180 seconds', value=execution.get('returncode', -1),
                               threshold=0, op='==', passed=False))
        if result is None:
            checks.append(dict(name='metrics.json produced',value=0,threshold=1,op='==',passed=False))
        passed = sum(c['passed'] for c in checks)
        total_failed += len(checks)-passed
        runtime = execution.get('runtime_seconds', result.get('runtime_seconds',0) if result else 0)
        figures = ', '.join(f'[{f}]({name}/{f})' for f in result.get('figures',[])) if result else ''
        lines.append(f'| {name} | {runtime:.2f} | {passed}/{len(checks)} | {status} | {figures} |')
        details += ['', f'## {name}', '', f'[Full metrics]({name}/metrics.json)', '',
                    '| Check | Value | Threshold | Result |', '|---|---:|---:|---|']
        for c in checks:
            details.append(f"| {c['name']} | {c['value']:.9g} | {c['op']} {c['threshold']:.9g} | {'PASS' if c['passed'] else 'FAIL'} |")
        if result and result.get('notes'):
            details += ['', *[f'- {note}' for note in result['notes']]]
    (OUT/'summary.md').write_text('\n'.join(lines+details)+'\n')
    return total_failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only', action='store_true', help='Summarize existing metrics without rerunning')
    parser.add_argument('--long', action='store_true', help='Also rerun the ~40 min long_turbulence test')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    executions = {}
    state_file = OUT/'executions.json'
    if args.report_only:
        executions = json.loads(state_file.read_text()) if state_file.exists() else {}
    else:
        previous = json.loads(state_file.read_text()) if state_file.exists() else {}
        # A skipped long test keeps its last recorded execution.
        executions = {k: v for k, v in previous.items() if k in LONG and not args.long}
        env = dict(os.environ, PYTHONPATH=str(ROOT), MPLCONFIGDIR='/tmp/phrike-dust-mpl',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        for name in TESTS:
            if name in LONG and not args.long:
                continue
            out = OUT/name
            out.mkdir(exist_ok=True)
            start = time.perf_counter()
            print(f'Running {name}', flush=True)
            # Remove a stale result only when its replacement campaign is starting.
            # This avoids mistaking a previous pass for a crashed/timeout run.
            (out/'metrics.json').unlink(missing_ok=True)
            with (out/'run.log').open('w') as log:
                try:
                    proc = subprocess.run([sys.executable, str(Path(__file__).with_name(name+'.py'))],
                                          cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                          timeout=3600 if name in LONG else 180)
                    status = 'completed' if proc.returncode in (0,1) and (out/'metrics.json').exists() else 'process failed'
                    executions[name] = dict(status=status, returncode=proc.returncode,
                                            runtime_seconds=time.perf_counter()-start)
                except subprocess.TimeoutExpired:
                    executions[name] = dict(status='timeout',returncode=-1,
                                            runtime_seconds=time.perf_counter()-start)
            # Retain checkpointed results if a later script fails or is interrupted.
            state_file.write_text(json.dumps(executions,indent=2)+'\n')
            summarize(executions)
            print(f"{'PASS' if executions[name]['returncode']==0 else 'FAIL'} {name}: "
                  f"exit={executions[name]['returncode']}, {executions[name]['runtime_seconds']:.2f}s",flush=True)
    failed = summarize(executions)
    print(f"{'PASS' if failed==0 else 'FAIL'} campaign: {failed} failed checks; {OUT/'summary.md'}")
    return int(failed > 0)


if __name__ == '__main__':
    raise SystemExit(main())
