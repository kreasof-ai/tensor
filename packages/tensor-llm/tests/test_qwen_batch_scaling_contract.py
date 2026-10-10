"""Stress trace nesting and qualification must retain every slot and commit."""
from pathlib import Path
import pytest
from benchmarks.qwen35.batch_scaling import workload,compare_records


def test_nested_batch_workloads_preserve_prompts_and_complete_load():
    path=Path(__file__).resolve().parents[3]/'docs/research/data/qwen35-native-h200/scaling/workload-c64.json.gz'
    small=workload(8,path)
    for slots in (16,32,64):
        large=workload(slots,path)
        assert large['requests'][:len(small['requests'])]==small['requests']
        assert len(large['requests'])==slots and not large['warmup']
        assert all(len(r['prompt_token_ids'])==32000 and r['output_tokens']==16000 for r in large['requests'])
        assert len({tuple(r['prompt_token_ids']) for r in large['requests']})==slots
        small=large


def test_batch_comparison_rejects_changed_slot_commit_or_prefix():
    rows=[dict(finite=True,kv_prefix_unchanged=True,commits=[dict(depth=3,position=32003)]) for _ in range(16)]
    assert compare_records(rows,rows)['passed']
    changed=[dict(r) for r in rows];changed[11]['commits']=[dict(depth=3,position=32002)]
    report=compare_records(rows,changed)
    assert not report['passed'] and report['differences']==[dict(slot=11,components=['commits'])]
    changed=[dict(r) for r in rows];changed[3]['kv_prefix_unchanged']=False
    assert not compare_records(rows,changed)['passed']
    with pytest.raises(ValueError,match='coverage'):compare_records(rows,rows[:8])
