"""Extend the existing digits experiment with isolated, balanced measurements."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import write_json, wilson
from .edge_protocol import Config, VARIANTS, choose_variant, schedule
from .measurement import RunRecord, environment, file_hash, peak_memory

LIMITATIONS = [
    'The 1,797 UCI 8x8 digit images are a small benchmark, not a representative production vision workload.',
    'Training seeds share one fixed test split. Repeated timings are not independent datasets or accuracy samples.',
    'Checkpoint and variant selection use validation only. Test results are reported after the decision is saved.',
    'CPU-only measurements on the recorded host; no verified edge-device or Windows deployment result.',
    'No instrumented power sensor: watts and joules are unavailable. No energy saving is inferred from latency.',
    'Peak memory is the OS whole-process high-water mark, including imports, construction and model loading; serialized bytes measure storage.',
    'Fresh workers and balanced positions reduce carryover bias, but background load, CPU frequency and temperature are not controlled.',
    'Linux, macOS and Windows peak-memory APIs have different accounting semantics; compare within the same OS/runtime.',
    'The declared budgets are experimental criteria, not customer requirements or measured customer impact.',
]


def configure_runtime(config):
    import torch
    torch.set_num_threads(config.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.backends.quantized.engine == 'none':
        raise RuntimeError('This PyTorch build has no quantized CPU engine')
    return {'torch': torch.__version__, 'intraop_threads': torch.get_num_threads(),
            'interop_threads': torch.get_num_interop_threads(),
            'quantized_engine': torch.backends.quantized.engine,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}


def train_worker(out, config, seed):
    import torch
    from .edge_vision import digits_data, train_variants
    runtime = configure_runtime(config)
    x, y, splits, digest = digits_data()
    if (out / 'splits.json').exists():
        if json.loads((out / 'splits.json').read_text()) != splits:
            raise ValueError('Dataset partitions changed between seeds')
    else:
        write_json(out / 'splits.json', splits)
    nets, histories = train_variants(x, y, splits, seed, config.epochs)
    artifacts = {}
    for name, net in nets.items():
        target = out / 'models' / f'{seed}_{name}.pt'
        target.parent.mkdir(exist_ok=True)
        with target.open('wb') as stream:
            torch.save(net.state_dict(), stream)
        artifacts[name] = {'path': str(target.relative_to(out)), 'sha256': file_hash(target),
                           'bytes': target.stat().st_size}
    write_json(out / f'training_seed_{seed}.json', {'seed': seed, 'dataset_sha256': digest,
               'runtime': runtime, 'histories': histories, 'artifacts': artifacts,
               'checkpoint_rule': 'minimum validation cross entropy, first epoch on ties'})


def load_variant(out, seed, variant):
    import torch
    from .edge_vision import model, quantize
    training = json.loads((out / f'training_seed_{seed}.json').read_text())
    artifact = training['artifacts'][variant]
    target = out / artifact['path']
    if file_hash(target) != artifact['sha256']:
        raise ValueError('Model artifact hash mismatch')
    started = time.perf_counter()
    net = model(128 if variant == 'teacher_fp32' else 24).eval()
    if variant == 'student_dynamic_int8':
        net = quantize(net)
    net.load_state_dict(torch.load(target, map_location='cpu', weights_only=True))
    net.eval()
    if variant == 'student_dynamic_int8':
        packed = [m for m in net.modules() if isinstance(m, torch.ao.nn.quantized.dynamic.Linear)]
        if len(packed) != 2 or any(m.weight().dtype != torch.qint8 for m in packed):
            raise RuntimeError('Expected two actual qint8 Linear layers')
    return net, artifact, training, time.perf_counter() - started


def quality(net, x, y, indices):
    from sklearn.metrics import confusion_matrix
    from .edge_vision import predict
    predictions = predict(net, x[indices])
    labels = y[indices].numpy()
    correct = int((predictions == labels).sum())
    return {'accuracy': correct / len(indices), 'correct': correct, 'total': len(indices),
            'accuracy_ci95': wilson(correct, len(indices)), 'indices': indices,
            'predictions': predictions.tolist(), 'labels': labels.tolist(),
            'confusion_matrix': confusion_matrix(labels, predictions, labels=list(range(10))).tolist()}


def evaluate_worker(out, config, seed, variant, trial, position, test=False):
    from .edge_vision import digits_data, measure
    runtime = configure_runtime(config)
    if test and not (out / 'selection.json').exists():
        raise ValueError('A saved selection is required before test evaluation')
    net, artifact, training, load_seconds = load_variant(out, seed, variant)
    x, y, splits, digest = digits_data()
    if digest != training['dataset_sha256'] or splits != json.loads((out / 'splits.json').read_text()):
        raise ValueError('Dataset or splits changed since training')
    indices = splits['test' if test else 'validation']
    result = {'seed': seed, 'variant': variant, 'pid': os.getpid(), 'runtime': runtime,
              'artifact_sha256': artifact['sha256'], 'dataset_sha256': digest,
              'serialized_state_bytes': artifact['bytes'], 'load_seconds': load_seconds}
    scored = quality(net, x, y, indices)
    if test:
        result.update({'test': scored, 'selection_sha256': file_hash(out / 'selection.json')})
        target = out / 'test' / f'{seed}_{variant}.json'
    else:
        result.update({'round': trial, 'position': position, 'validation': scored,
                       'validation_accuracy': scored['accuracy'],
                       'latency_batch1': measure(net, x[indices[:1]], config.repetitions),
                       'latency_batch32': measure(net, x[indices[:32]], config.repetitions)})
        result['peak_memory'] = peak_memory()
        target = out / 'trials' / f'{seed}_{trial}_{variant}.json'
    write_json(target, result)


def launch_worker(out, config, phase, seed, variant=None, trial=0, position=0):
    command = [sys.executable, '-m', 'portfolio.edge_benchmark', '--out', str(out),
               '--worker', phase, '--seed', str(seed), '--trial', str(trial), '--position', str(position)]
    if variant:
        command += ['--variant', variant]
    # Only runtime controls are passed explicitly; no credentials or user environment are recorded/copied.
    runtime_env = {name: str(config.threads) for name in
                   ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
    runtime_env['PYTHONUTF8'] = '1'
    if sys.platform == 'win32':
        runtime_env['SystemRoot'] = os.environ.get('SystemRoot', 'C:\\Windows')
    label = f'{phase}_{seed}_{trial}_{variant or "all"}'
    logs = out / 'logs'; logs.mkdir(exist_ok=True)
    write_json(logs / f'{label}.command.json', {'argv': command, 'runtime_controls': runtime_env,
                                             'timeout_seconds': 600})
    started = time.perf_counter()
    with (logs / f'{label}.txt').open('w') as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=runtime_env, timeout=600, check=False)
    write_json(logs / f'{label}.exit.json', {'returncode': completed.returncode,
                                          'elapsed_seconds': time.perf_counter() - started})
    completed.check_returncode()


def report(summary):
    selected = summary['selection']
    lines = ['# Isolated digits benchmark', '',
             f"Selected variant: **{selected['selected_variant'] or 'none'}** (validation-only decision).",
             '', selected['rule'], '', selected['recommendation'], '',
             '| Variant | Validation mean | Test mean | State KiB max | Peak MiB max | Batch-1 p50 median ms | Batch-1 p95 max ms | Gate |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for candidate in selected['candidates']:
        tests = [r['test']['accuracy'] for r in summary['test_rows'] if r['variant'] == candidate['variant']]
        peak = candidate['peak_bytes_max']
        peak_text = f'{peak / 2**20:.1f}' if peak is not None else 'unavailable'
        lines.append(f"| {candidate['variant']} | {candidate['validation_accuracy_mean']:.2%} | {sum(tests)/len(tests):.2%} | "
                     f"{candidate['state_bytes_max']/1024:.1f} | {peak_text} | {candidate['batch1_p50_ms_median']:.5f} | "
                     f"{candidate['batch1_p95_ms_max']:.5f} | {'eligible' if candidate['eligible'] else '; '.join(candidate['reasons'])} |")
    lines += ['', 'Test outcomes describe the frozen decision and do not change it. Inspect raw trial records before interpreting small timing differences.',
              '', '## Measurement scope', '', f"Host CPU: {summary['environment']['cpu']}. OS: {summary['environment']['os']}.",
              f"Power condition: {summary['environment']['power_condition']}; power and energy: unavailable.",
              '', '## Limitations', ''] + ['- ' + text for text in LIMITATIONS]
    return '\n'.join(lines) + '\n'


def run(out, config=Config()):
    config.validate()
    out = Path(out).resolve()
    sources = [Path(__file__).with_name(name) for name in
               ('edge_benchmark.py', 'edge_protocol.py', 'edge_vision.py', 'measurement.py', 'common.py')]
    host = environment(config.device_label, config.power_condition)
    protocol = {'experiment': 'digits-isolated-v2', 'config': config.as_dict(),
                'environment': host, 'schedule': schedule(config), 'argv': sys.argv,
                'warmup': 10, 'batch_sizes': [1, 32], 'selection_before_test': True,
                'timing_scope': 'warm CPU forward pass only; excludes preprocessing, load and process startup'}
    with RunRecord(out, protocol, sources):
        for seed in config.seeds:
            launch_worker(out, config, 'train', seed)
        rows = []
        for item in schedule(config):
            launch_worker(out, config, 'measure', item['seed'], item['variant'], item['round'], item['position'])
            rows.append(json.loads((out / 'trials' / f"{item['seed']}_{item['round']}_{item['variant']}.json").read_text()))
        selection = choose_variant(rows, config)
        write_json(out / 'selection.json', selection)
        test_rows = []
        for seed in config.seeds:
            for variant in VARIANTS:
                launch_worker(out, config, 'test', seed, variant)
                test_rows.append(json.loads((out / 'test' / f'{seed}_{variant}.json').read_text()))
        summary = {'experiment': 'digits-isolated-v2', 'environment': host, 'config': config.as_dict(),
                   'selection': selection, 'rows': rows, 'test_rows': test_rows, 'limitations': LIMITATIONS}
        write_json(out / 'summary.json', summary)
        (out / 'REPORT.md').write_text(report(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/edge_isolated')
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 19, 43])
    for name, default in [('epochs', 100), ('rounds', 4), ('repetitions', 100), ('threads', 1)]:
        parser.add_argument('--' + name, type=int, default=default)
    for name, default in [('min-accuracy', .95), ('max-accuracy-drop', .02), ('max-state-kib', 16),
                          ('max-p95-ms', 1), ('max-peak-mib', 1024)]:
        parser.add_argument('--' + name, type=float, default=default)
    parser.add_argument('--device-label', default='development-host')
    parser.add_argument('--power-condition', choices=['unknown', 'ac', 'battery'], default='unknown')
    parser.add_argument('--worker', choices=['train', 'measure', 'test'], help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--variant', choices=VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument('--trial', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--position', type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.worker:
            out = Path(args.out)
            data = json.loads((out / 'protocol.json').read_text())['config']
            data['seeds'] = tuple(data['seeds'])
            config = Config(**data).validate()
            if args.seed not in config.seeds:
                raise ValueError('Worker seed not in protocol')
            if args.worker == 'train': train_worker(out, config, args.seed)
            else: evaluate_worker(out, config, args.seed, args.variant, args.trial, args.position, args.worker == 'test')
        else:
            config = Config(**{key: tuple(value) if key == 'seeds' else value for key, value in vars(args).items()
                               if key in Config.__dataclass_fields__}).validate()
            result = run(args.out, config)
            print(json.dumps({'status': 'complete', 'selected_variant': result['selection']['selected_variant']}))
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
