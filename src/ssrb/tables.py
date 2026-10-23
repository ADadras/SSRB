"""Aggregate runs using the accepted averaging order for visible choices."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np


def _mean_metrics(rows):
    metadata={'endpoint','visible_seed','image_id','source_id'}
    return {name:float(np.mean([r[name] for r in rows if r.get(name) is not None]))
            for name in set(rows[0])-metadata if any(r.get(name) is not None for r in rows)}


def aggregate(evaluations, output, status, optimizer_seeds=(0, 1, 2), averaging='visible-first'):
    entries=json.loads(Path(evaluations).read_text())
    if not entries:
        raise ValueError('no evaluations supplied')
    expected_seeds=set(optimizer_seeds)
    if not expected_seeds or len(expected_seeds)!=len(optimizer_seeds) or min(expected_seeds)<0:
        raise ValueError('declare a nonempty set of distinct optimizer seeds')
    if averaging not in {'visible-first','run-means'}:
        raise ValueError('unknown averaging rule')
    groups=defaultdict(list)
    visible=defaultdict(set)
    cells=defaultdict(set)
    run_means=defaultdict(list)
    for entry in entries:
        rows=json.loads((Path(entry['evaluation'])/'per_image.json').read_text())
        optimizer_seed=int(entry['optimizer_seed'])
        if optimizer_seed not in expected_seeds:
            raise ValueError('evaluation has an undeclared optimizer seed')
        entry_cells={(row['endpoint'],int(row['visible_seed'])) for row in rows}
        if not entry_cells or len({v for _,v in entry_cells})!=1:
            raise ValueError('each evaluation must contain one visible choice')
        for endpoint,choice in entry_cells:
            cell=(optimizer_seed,choice)
            if cell in cells[endpoint]:
                raise ValueError('duplicate endpoint/optimizer/visible cell')
            cells[endpoint].add(cell)
            run_means[endpoint].append(_mean_metrics([r for r in rows if r['endpoint']==endpoint]))
        for row in rows:
            key=(row['endpoint'],optimizer_seed,row['image_id'],row['source_id'])
            if row['visible_seed'] in visible[key]:
                raise ValueError('duplicate visible membership in analysis units')
            visible[key].add(row['visible_seed'])
            groups[key].append(row)
    by_endpoint=defaultdict(list)
    for key,rows in sorted(groups.items()):
        metrics=_mean_metrics(rows)
        by_endpoint[key[0]].append((key[1],metrics))
    result=[]
    for endpoint,rows in sorted(by_endpoint.items()):
        expected_cells={(seed,choice) for seed in expected_seeds for choice in (0,1,2)}
        if cells[endpoint]!=expected_cells:
            raise ValueError('paper aggregate requires every declared optimizer seed and all three visible choices')
        units=[r for _,r in rows] if averaging=='visible-first' else run_means[endpoint]
        keys=sorted({k for r in units for k in r})
        result.append(dict(endpoint=endpoint,status=status,averaging=averaging,analysis_units=len(units),
                           **{k:float(np.mean([r[k] for r in units if k in r])) for k in keys}))
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sorted({k for r in result for k in r}))
        writer.writeheader();writer.writerows(result)
    return result
