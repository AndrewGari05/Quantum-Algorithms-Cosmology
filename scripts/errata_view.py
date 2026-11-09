"""Corrected view of the single-seed campaign: errata re-runs replace the invalid rows (reads CSVs only).

Usage:
    python scripts/errata_view.py OLD_CAMPAIGN ERRATA_RESULTS --out NEW_CAMPAIGN

OLD_CAMPAIGN is a copy of hpc_20260907_130425 (seed 42). ERRATA_RESULTS is the
``results`` folder of the errata re-runs (thesis-errata 2f98c17):

  errata_fb_samplers_2p/_3p, errata_fb_genetic_2p/_3p   FakeBrisbane cells (HPC-3)
  errata_qm1_ideal / _counts / _noisy                   QMCMC rungs (QM-1)

The output is a campaign folder with the usual task names
(``<family>_<model>_<nqpp|nb><g>_noise-<level>/model_<model>/results_config.csv``)
that every plotting script reads as before:

* every FakeBrisbane task of the old campaign is dropped; the re-run FakeBrisbane
  tasks are added (the re-runs wrote no noise tag in the folder name, so it is
  taken from their ``noise`` column);
* in every sampler task the old ``QMCMC 50%`` and ``QMCMC 100%`` rows are dropped
  and the QM-1 re-run rows of the same model, nqpp and level are put in their
  place (``none`` from errata_qm1_ideal, ``none-counts`` from errata_qm1_counts,
  ``readout`` / ``full`` from errata_qm1_noisy); the re-run ``Classical MCMC`` row
  must equal the old one, which is checked;
* re-run cells with no old task (e.g. QMCMC at larger nqpp) are added as tasks of
  their own;
* nothing else changes: every other row is copied as it was.

A report of what was replaced, dropped and added is printed and written to
``errata_view_report.txt`` in the output folder.
"""
import argparse
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import campaign_io  # noqa: E402

QMCMC = ('QMCMC 50%', 'QMCMC 100%')
QM1_SOURCES = {'none': 'errata_qm1_ideal', 'none-counts': 'errata_qm1_counts',
               'readout': 'errata_qm1_noisy', 'full': 'errata_qm1_noisy'}
FB_SOURCES = ('errata_fb_samplers_2p', 'errata_fb_samplers_3p',
              'errata_fb_genetic_2p', 'errata_fb_genetic_3p')
COMPARE = ('Om_mean', 'Om_std', 'H0_mean', 'H0_std', 'acceptance', 'ESS')


def _rows(path):
    with open(path, newline='') as fh:
        return list(csv.DictReader(fh))


def _task_rows(task_dir):
    out = []
    for p in campaign_io.result_csvs(task_dir):
        out += _rows(p)
    return out


def _grid_tag(fam):
    return 'nqpp' if fam == 'samplers' else 'nb'


def _key(fam, mod, g, noi):
    return f'{fam}_{mod}_{_grid_tag(fam)}{g}_noise-{noi}'


def _noise_level(row, folder_noise):
    """Level of a re-run row: folder tag, else the noise column, with the counts route."""
    noi = folder_noise or (row.get('noise') or '').strip() or 'none'
    if noi == 'none' and (row.get('proposal_route') or '').strip() == 'counts':
        return 'none-counts'
    return noi


def load_errata(root):
    """``{task_key: [rows]}`` of every re-run, keyed like the old campaign."""
    out = {}
    for sub in sorted(set(QM1_SOURCES.values())) + list(FB_SOURCES):
        base = os.path.join(root, sub)
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            tags = campaign_io.parse_task_name(name)
            if tags is None:
                continue
            for r in _task_rows(os.path.join(base, name)):
                noi = _noise_level(r, tags['noi'])
                k = _key(tags['fam'], tags['mod'], tags['g'], noi)
                r['_source'] = f'{sub}/{name}'
                out.setdefault(k, []).append(r)
    return out


def write_task(out_root, key, mod, rows):
    d = os.path.join(out_root, key, f'model_{mod}')
    os.makedirs(d, exist_ok=True)
    cols = []
    for r in rows:
        cols += [c for c in r if not c.startswith('_') and c not in cols]
    with open(os.path.join(d, campaign_io.RESULT_CSV), 'w', newline='') as fh:
        w = csv.DictWriter(fh, cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build(old_root, errata_root, out_root):
    errata = load_errata(errata_root)
    used, report = set(), []
    os.makedirs(out_root, exist_ok=True)
    for name in sorted(os.listdir(old_root)):
        tags = campaign_io.parse_task_name(name)
        task_dir = os.path.join(old_root, name)
        if tags is None or not os.path.isdir(task_dir):
            continue
        rows = _task_rows(task_dir)
        if not rows:
            continue
        noi = tags['noi'] or 'none'
        key = _key(tags['fam'], tags['mod'], tags['g'], noi)
        if noi == 'fakebrisbane':
            if key in errata:
                write_task(out_root, key, tags['mod'], errata[key])
                used.add(key)
                report.append(f'replaced  {key}: FakeBrisbane re-run ({errata[key][0]["_source"]})')
            else:
                report.append(f'dropped   {key}: old FakeBrisbane cell, no re-run (HPC-3)')
            continue
        if tags['fam'] == 'samplers' and any(r['Method'] in QMCMC for r in rows):
            new = [r for r in errata.get(key, []) if r['Method'] in QMCMC]
            keep = [r for r in rows if r['Method'] not in QMCMC]
            if new:
                old_mc = next((r for r in rows if r['Method'] == 'Classical MCMC'), None)
                new_mc = next((r for r in errata[key] if r['Method'] == 'Classical MCMC'), None)
                same = old_mc is not None and new_mc is not None and all(
                    (old_mc.get(c) or '') == (new_mc.get(c) or '') for c in COMPARE)
                rows = keep + new
                used.add(key)
                report.append(f'replaced  {key}: QMCMC 50%/100% from {new[0]["_source"]}'
                              f' (Classical MCMC identical: {same})')
            else:
                rows = keep
                report.append(f'dropped   {key}: old QMCMC rows, no QM-1 re-run of this cell')
        write_task(out_root, key, tags['mod'], rows)
    for key in sorted(set(errata) - used):
        mod = campaign_io.parse_task_name(key)['mod']
        write_task(out_root, key, mod, errata[key])
        report.append(f'added     {key}: re-run cell with no old task ({errata[key][0]["_source"]})')
    text = '\n'.join(report) + '\n'
    with open(os.path.join(out_root, 'errata_view_report.txt'), 'w', encoding='utf-8') as fh:
        fh.write(text)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('old_campaign')
    ap.add_argument('errata_results')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    report = build(a.old_campaign, a.errata_results, a.out)
    for line in report:
        print(' ', line)
    kinds = {k: sum(1 for x in report if x.startswith(k)) for k in ('replaced', 'dropped', 'added')}
    print(f'  {kinds} -> {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
