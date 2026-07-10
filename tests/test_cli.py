"""
End-to-end CLI smoke tests: every subcommand and every sweep mode is invoked as
a subprocess against the synthetic trajectory, and the primary output files are
checked for existence and loadability.
"""

import sys
import subprocess

import numpy as np
import pytest


def _run(args, cwd):
    proc = subprocess.run([sys.executable, '-m', 'BRMSD.cli'] + args,
                          cwd=cwd, capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"command failed: {' '.join(args)}\nSTDERR:\n{proc.stderr[-2000:]}")
    return proc


# ~all frames of the 50-frame fixture; small maxiter keeps it fast
FAST = ['-max', '20', '-tol', '1e-3']


def test_cli_align(traj, tmp_path):
    d = tmp_path
    _run(['align', '-s', traj['gro'], '-f', traj['xtc'], '-sel', 'name CA',
          '-sg', '0.5', '-p', 'avg.pdb', '-rmsd', 'rmsd.dat', '-rmsf', 'rmsf.dat',
          '-wt', 'w.dat', '-g', 'a.log', '-rot', 'rot.dat', '-rfmt', 'quaternion']
         + FAST, cwd=d)
    assert (d / 'avg.pdb').exists()
    assert (d / 'w.dat').exists()
    rot = np.loadtxt(d / 'rot.dat', comments='#')     # exercises the scipy path
    assert rot.shape[1] == 5                          # time + 4 quaternion cols


def test_cli_smooth(traj, tmp_path):
    d = tmp_path
    _run(['smooth', '-s', traj['gro'], '-f', traj['xtc'], '-sel', 'name CA',
          '-sg', '0.5', '-W', '3', '-o', 'sm.xtc', '-rmsd', 'r.dat',
          '-rmsf', 'rf.dat', '-wt', 'w.dat', '-g', 's.log', '-j', '2'] + FAST, cwd=d)
    assert (d / 'sm.xtc').exists()
    assert (d / 'rf.dat').exists()


def test_cli_cluster(traj, tmp_path):
    d = tmp_path
    _run(['cluster', '-s', traj['gro'], '-f', traj['xtc'], '-sel', 'name CA',
          '-A', '2', '-sg', '0.3', '-tau', '2.0', '-nr', '2', '-p', 'cc.pdb',
          '-rmsd', 'r.dat', '-rmsf', 'rf.dat', '-wt', 'w.dat', '-q', 'q.dat',
          '-l', 'lab.dat', '-pop', 'pop.dat', '-g', 'c.log', '-j', '2'] + FAST, cwd=d)
    assert (d / 'cc.pdb').exists()
    assert (d / 'pop.dat').exists()
    # responsibilities.dat is an xmgrace multi-dataset file (&-separated blocks)
    assert (d / 'q.dat').stat().st_size > 0


def test_cli_domain_sequential(traj, tmp_path):
    d = tmp_path
    _run(['domain', '-s', traj['gro'], '-f', traj['xtc'], '-sel', 'name CA',
          '-sg', '1.0', '-nd', '2', '-nr', '2', '-p', 'dr.pdb', '-wt', 'w.dat',
          '-d', 'dom.dat', '-g', 'd.log'] + FAST, cwd=d)
    assert (d / 'dr.pdb').exists()
    assert (d / 'dom.dat').exists()


def test_cli_domain_simultaneous(traj, tmp_path):
    d = tmp_path
    _run(['domain', '-s', traj['gro'], '-f', traj['xtc'], '-sel', 'name CA',
          '-pm', 'simultaneous', '-mu', '1.0', '-sg', '1.0', '-nd', '2', '-nr', '2',
          '-p', 'dr.pdb', '-wt', 'w.dat', '-d', 'dom.dat', '-g', 'd.log'] + FAST, cwd=d)
    assert (d / 'dr.pdb').exists()


def test_cli_focus(traj, tmp_path):
    d = tmp_path
    _run(['focus', '-s', traj['gro'], '-f', traj['xtc'], '-sel', 'name CA',
          '-dom', traj['domainB_sel'], '-sg', '0.5', '-mr', '1.0', '-p', 'af.pdb',
          '-rmsd', 'r.dat', '-rmsf', 'rf.dat', '-wt', 'w.dat', '-g', 'f.log'] + FAST, cwd=d)
    assert (d / 'af.pdb').exists()
    assert (d / 'w.dat').exists()


@pytest.mark.parametrize('args,out', [
    (['-m', 'align',   '--mode', 'sigma',    '--range', '0.2', '2.0', '4', '--log-space'], 'align_sigma_sweep.csv'),
    (['-m', 'cluster', '--mode', 'sigma',    '--range', '0.2', '1.0', '3', '--sigma', '0.3', '--tau', '2.0', '-K', '2'], 'cluster_sigma_sweep.csv'),
    (['-m', 'cluster', '--mode', 'tau',      '--range', '1.0', '5.0', '3', '--sigma', '0.3', '-K', '2'], 'cluster_tau_sweep.csv'),
    (['-m', 'cluster', '--mode', 'K',        '--range', '2', '4', '3', '--sigma', '0.3', '--tau', '2.0'], 'cluster_K_sweep.csv'),
    (['-m', 'focus',   '--mode', 'mu',       '--range', '0.0', '0.9', '3', '--sigma', '0.5'], 'focus_mu_sweep.csv'),
    (['-m', 'domain',  '--mode', 'sigma',    '--range', '0.5', '3.0', '3', '--n-domains', '2'], 'domain_sigma_sweep.csv'),
    (['-m', 'domain',  '--mode', 'K',        '--range', '2', '4', '3', '--sigma', '1.0'], 'domain_K_sweep.csv'),
    (['-m', 'domain',  '--mode', 'mask_thr', '--range', '0.1', '0.9', '4', '--sigma', '1.0', '--n-domains', '2'], 'domain_mask_thr_sweep.csv'),
    (['-m', 'smooth',  '--mode', 'sigma',    '--range', '0.2', '2.0', '3', '-W', '3'], 'smooth_sigma_sweep.csv'),
    (['-m', 'smooth',  '--mode', 'window',   '--range', '2', '6', '3', '--sigma', '0.5'], 'smooth_window_sweep.csv'),
])
def test_cli_sweep(traj, tmp_path, args, out):
    d = tmp_path
    extra = ['-nr', '1']
    if '-m' in args and args[args.index('-m') + 1] == 'focus':
        extra += ['-dom', traj['domainB_sel']]
    _run(['sweep'] + args + ['-s', traj['gro'], '-f', traj['xtc'],
          '-sel', 'name CA', '-g', 'sweep.log'] + FAST + extra, cwd=d)
    csv = d / out
    assert csv.exists(), f"missing sweep CSV {out}"
    with open(csv) as fh:
        lines = fh.read().strip().splitlines()
    assert len(lines) >= 2      # header + at least one row


def test_cli_integer_sweep_honours_n(traj, tmp_path):
    """domain/K --range 2 4 3 must yield exactly K = 2, 3, 4 (three rows)."""
    d = tmp_path
    _run(['sweep', '-m', 'domain', '--mode', 'K', '--range', '2', '4', '3',
          '--sigma', '1.0', '-s', traj['gro'], '-f', traj['xtc'],
          '-sel', 'name CA', '-nr', '1', '-o', 'dk.csv', '-g', 'sw.log'] + FAST, cwd=d)
    with open(d / 'dk.csv') as fh:
        rows = fh.read().strip().splitlines()[1:]
    ks = [int(r.split(',')[0]) for r in rows]
    assert ks == [2, 3, 4]
