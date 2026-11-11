from pathlib import Path
import csv
import math
import statistics
from collections import defaultdict

ROOT = Path(__file__).resolve().parent


def read(name):
    with (ROOT/name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def close(a, b, tolerance=1e-9):
    assert math.isclose(float(a),float(b),rel_tol=0,abs_tol=tolerance),(a,b)


def verify_distillation():
    metrics=[('prediction_accuracy','unit_correct','unit_total'),
             ('cross_view_recall','visual_correct','retrieval_total'),
             ('cross_modal_recall','audio_correct','retrieval_total')]
    counts=defaultdict(lambda:defaultdict(int))
    ids=defaultdict(set)
    for row in read('distillation/per_view_counts.csv'):
        key=(row['configuration'],int(row['seed']))
        ids[key].add((row['utterance_id'],row['view_index']))
        for _,a,b in metrics:
            # Shared retrieval denominators must only be accumulated once.
            counts[key][a]+=int(row[a])
        counts[key]['unit_total']+=int(row['unit_total'])
        counts[key]['retrieval_total']+=int(row['retrieval_total'])
    assert len(counts)==6 and all(len(v)==1440 for v in ids.values())
    values=defaultdict(list)
    for row in read('distillation/per_seed_metrics.csv'):
        key=(row['configuration'],int(row['seed']))
        for name,a,b in metrics:
            score=100*counts[key][a]/counts[key][b]
            close(score,row[name+'_pct'])
            values[row['configuration'],name].append(score)
    for row in read('distillation/summary.csv'):
        vals=values[row['configuration'],row['metric']]
        close(statistics.mean(vals),row['mean_pct'])
        close(statistics.stdev(vals),row['sample_std_pct'])
    split_counts=defaultdict(int)
    speakers=defaultdict(set)
    for row in read('distillation/dataset_splits.csv'):
        split_counts[row['split']]+=1
        speakers[row['split']].add(row['speaker_id'])
    assert dict(split_counts)=={'train':840,'valid':360,'test':360}
    assert not (speakers['train'] & speakers['valid'] or speakers['train'] & speakers['test'] or speakers['valid'] & speakers['test'])
    print('Agg2S: 6 runs, 360 test utterances, all three metrics reproduced.')


def verify_tcd():
    groups=defaultdict(lambda:defaultdict(int))
    ids=defaultdict(set)
    for row in read('tcd_baseline/per_video_counts.csv'):
        hours=int(row['fine_tuning_hours'])
        assert row['utterance_id'] not in ids[hours]
        ids[hours].add(row['utterance_id'])
        for angle in [row['yaw_degrees'],'overall']:
            g=groups[hours,angle]
            g['videos']+=1
            for col in ['substitutions','deletions','insertions','reference_words']:
                g[col]+=int(row[col])
    assert all(len(v)==11564 for v in ids.values()) and ids[30]==ids[433]
    for row in read('tcd_baseline/summary.csv'):
        g=groups[int(row['fine_tuning_hours']),row['yaw_degrees']]
        for col,val in g.items(): assert val==int(row[col])
        close(100*(g['substitutions']+g['deletions']+g['insertions'])/g['reference_words'],row['wer_pct'])
    print('TCD-TIMIT: 23,128 video evaluations; both checkpoint WER summaries reproduced.')


def verify_grading():
    counts=defaultdict(lambda:[0,0])
    for row in read('decision_grading/test_logits.csv'):
        pred=max(range(4),key=lambda i:float(row[f'logit_{i}']))
        key=row['configuration'],row['seed'],row['dataset']
        counts[key][0]+=int(pred==int(row['target_grade']))
        counts[key][1]+=1
    metrics=defaultdict(list)
    for row in read('decision_grading/per_run_metrics.csv'):
        c,n=counts[row['configuration'],row['seed'],row['dataset']]
        assert (c,n)==(int(row['correct']),int(row['total']))
        close(100*c/n,row['accuracy_pct'])
        metrics[row['configuration'],row['dataset']].append(100*c/n)
    for row in read('decision_grading/summary.csv'):
        vals=metrics[row['configuration'],row['dataset']]
        close(statistics.mean(vals),row['mean_accuracy_pct'])
        if len(vals)>1: close(statistics.stdev(vals),row['sample_std_pct'])
    print('Decision grading: all 12 Supplementary Table S1 cells reproduced from logits.')


def verify_representation():
    units=read('representation/unit_consistency.csv')
    views=read('representation/view_consistency.csv')
    assert len(units)==447 and len(views)==360
    for row in units:
        n=int(row['feature_count'])
        for model in ['baseline','muvirl']:
            close(1-n/(n-1)*float(row[f'{model}_squared_distance']),row[f'{model}_similarity'])
    improved_units=sum(float(r['muvirl_similarity'])>float(r['baseline_similarity']) for r in units)
    improved_views=sum(float(r['muvirl_similarity'])>float(r['baseline_similarity']) for r in views)
    assert (improved_units,improved_views)==(436,353)
    diagonals=defaultdict(list)
    for row in read('representation/temporal_similarity.csv'):
        offset=int(row['candidate_frame'])-int(row['query_frame'])
        diagonals[row['pair_type'],offset].append(float(row['cosine_similarity']))
    for row in read('representation/temporal_offsets.csv'):
        vals=diagonals[row['pair_type'],int(row['frame_offset'])]
        assert len(vals)==int(row['frame_pairs'])
        close(statistics.mean(vals),row['mean_similarity'])
    print(f'Figure 6: {improved_units}/447 units and {improved_views}/360 utterances; frame-offset profiles reproduced.')


if __name__=='__main__':
    verify_distillation()
    verify_tcd()
    verify_grading()
    verify_representation()
    print('All released result summaries verified.')
