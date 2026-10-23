"""Sample CIMD while restricting reads to test images, its checkpoint and pinned upstream."""
import argparse
import atexit
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile



def finish_output(generated, output):
    # Landlock can reject reparenting even within the output tree. Copying the
    # completed bytes and then unlinking the old names preserves the bank.
    for name in ('predictions.pt', 'frozen.json'):
        if (output/name).exists():
            raise FileExistsError(output/name)
        shutil.copyfile(generated/name, output/name)
        (generated/name).unlink()
    generated.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cimd-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--sampling-seed', type=int, default=3490)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--check-denied-read', action='append', type=Path, default=[])
    args = parser.parse_args()
    root, manifest, output = args.root.resolve(), args.manifest.resolve(), args.output.resolve()
    if not manifest.is_relative_to(root):
        raise ValueError('the image manifest must be inside the image root')
    if output.exists():
        raise FileExistsError(output)
    upstream = args.cimd_root.resolve()
    output.mkdir(parents=True)
    temp_parent = '/var/tmp' if os.access('/var/tmp', os.W_OK) else '/tmp'
    scratch = Path(tempfile.mkdtemp(prefix='ssrb-cimd-', dir=temp_parent))
    probe = Path(tempfile.mkdtemp(prefix='ssrb-cimd-denied-', dir=temp_parent))
    control = probe/'control.txt'
    control.write_text('This file must become unreadable before training.\n')
    owner = os.getpid()
    atexit.register(lambda: shutil.rmtree(scratch, ignore_errors=True) if os.getpid() == owner else None)
    os.environ.update(TMPDIR=str(scratch), CUDA_CACHE_PATH=str(scratch/'cuda'),
                      MPLCONFIGDIR=str(scratch/'mpl'), GIT_CONFIG_GLOBAL='/dev/null',
                      GIT_CONFIG_SYSTEM='/dev/null', GIT_ATTR_NOSYSTEM='1')
    tempfile.tempdir = str(scratch)
    # Import runtime modules after choosing scratch, so their temporary files
    # also remain readable during cleanup under the same restriction.
    from ssrb.cimd import load_official_modules
    from ssrb.comparisons import infer
    import torch
    from ssrb.isolation import restrict_reads
    import ssrb
    load_official_modules(upstream)
    paths = [root, output, scratch, upstream, args.checkpoint.resolve(), sys.prefix, sys.base_prefix,
             Path(ssrb.__file__).resolve().parent, '/usr', '/lib', '/lib64',
             '/etc', '/dev', '/proc', '/sys']
    abi = restrict_reads(paths)
    try:
        control.read_bytes()
    except PermissionError:
        pass
    else:
        raise RuntimeError('the read restriction did not reject its control file')
    control.unlink()
    probe.rmdir()
    for denied in args.check_denied_read:
        try:
            with denied.open('rb') as stream:
                stream.read(1)
        except PermissionError:
            pass
        else:
            raise RuntimeError('an additional read check was not rejected')
    (output/'read_isolation.json').write_text(json.dumps(dict(
        landlock_abi=abi, control_read_denied=True, additional_denied_reads=len(args.check_denied_read),
        scope='File reads restricted to test images, checkpoint, output, scratch, runtime and pinned upstream.'), indent=2)+'\n')
    info = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if info.get('method') != 'cimd':
        raise ValueError('this runner requires a CIMD checkpoint')
    del info
    generated = output/'bank'
    infer(args.checkpoint, root, manifest, generated, cimd_root=upstream,
          device=args.device, deterministic=args.deterministic,
          batch_size=args.batch_size, sampling_seed=args.sampling_seed)
    finish_output(generated, output)


if __name__ == '__main__':
    main()
