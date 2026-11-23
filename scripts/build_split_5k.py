"""
Draw the 5,000-frame experiment set and write data/cache/set_dev.csv and set_test.csv.

Keeps the 1,076 frames of the August scaled run, then adds new frames from the manifest
so every survey year x density band cell holds about the same number (242). Frames in
the benchmark, the hand labels or the 28-frame fresh check are left out. 1,000 of the
new frames are sealed as the test set, proportionally per cell. `weight` is the cell's
corpus size over its sample size, for corpus-weighted figures.

Needs data/cache/manifest.csv, benchmark.csv, fresh.csv and the August run's
results/dataset_scaled/chunks/. The committed set_*.csv files are its output.
"""
import glob, os, sys, pandas as pd, numpy as np
sys.path.insert(0, '.')
from scripts.build_benchmark import per_screenshot, BANDS
name = lambda k: os.path.splitext(os.path.basename(k))[0]
per = per_screenshot(pd.read_csv('data/cache/manifest.csv', low_memory=False))
per['frame'] = per.screenshot_key.map(name)
per = per.drop_duplicates('frame')
per['band'] = per.dots.map(lambda d: next((b for b, lo, hi in BANDS if lo <= d <= hi), None))
per = per[per.band.notna()]
old = set(pd.concat([pd.read_csv(p) for p in glob.glob('results/dataset_scaled/chunks/*_frames.csv')]).frame)
excl = (set(pd.read_csv('data/cache/benchmark.csv').screenshot_key.map(name))
        | {os.path.splitext(f)[0] for f in os.listdir('data/labels')}
        | set(pd.read_csv('data/cache/fresh.csv').screenshot_key.map(name))) - old
oldf = per[per.frame.isin(old)].assign(origin='scaled_1076')
pool = per[~per.frame.isin(old | excl)]
cells = sorted(set(zip(per.year, per.band)))
inc = lambda df, c: (df.year == c[0]) & (df.band == c[1])
need = 5000 - len(oldf)
have = {c: inc(oldf, c).sum() for c in cells}
avail = {c: inc(pool, c).sum() for c in cells}
lo, hi = 0, 5000                      # equal per-cell total T, water-filled under availability
while lo < hi:
    T = (lo + hi + 1) // 2
    lo, hi = (T, hi) if sum(min(avail[c], max(0, T - have[c])) for c in cells) <= need else (lo, T - 1)
quota = {c: min(avail[c], max(0, lo - have[c])) for c in cells}
short = need - sum(quota.values())
for c in sorted(cells, key=lambda c: -(avail[c] - quota[c])):
    if short <= 0: break
    if avail[c] > quota[c]: quota[c] += 1; short -= 1
picks = []
for c, q in quota.items():
    cell = pool[inc(pool, c)].sample(frac=1, random_state=2026)
    cell = cell.assign(_r=cell.groupby('colony').cumcount()).sort_values('_r', kind='stable').drop(columns='_r')
    picks.append(cell.head(q))
new = pd.concat(picks).assign(origin='new_2026-11').sample(frac=1, random_state=7)
new['_k'] = new.groupby(['year', 'band']).cumcount()
new['_n'] = new.groupby(['year', 'band']).frame.transform('size')
test = new[new._k < (new._n * 1000 / len(new)).round()]
test = test.head(1000) if len(test) >= 1000 else pd.concat([test, new[~new.frame.isin(test.frame)].head(1000 - len(test))])
dev = pd.concat([oldf, new[~new.frame.isin(test.frame)]])
allf = pd.concat([dev.assign(split='dev'), test.assign(split='test')])
cs = per.groupby(['year', 'band']).size()
ss = allf.groupby(['year', 'band']).size()
allf['weight'] = [cs[(y, b)] / ss[(y, b)] for y, b in zip(allf.year, allf.band)]   # corpus reweighting
assert allf.frame.is_unique and not allf.frame.isin(excl).any() and len(allf) == 5000
cols = ['screenshot_key', 'highres_key', 'dots', 'birds', 'nests', 'species', 'year', 'colony',
        'region', 'band', 'origin', 'split', 'weight']
for s, g in allf.groupby('split'): g[cols].to_csv(f'data/cache/set_{s}.csv', index=False)
print('dev', len(dev), 'test', len(test), '| per-cell total', lo)
print(pd.crosstab(allf.year, [allf.split, allf.band]))
print(allf.groupby('split').agg(frames=('frame', 'size'), dots=('dots', 'sum'), colonies=('colony', 'nunique')))
