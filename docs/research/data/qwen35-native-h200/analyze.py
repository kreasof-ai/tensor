import json
from pathlib import Path
root=Path(__file__).resolve().parent
rows=[]
for name in ('tensor-h200-ar-c8','tensor-h200-mtp-lookup-c8'):
    report=json.loads((root/name/'report.json').read_text())
    server=report['servers'][0]
    point=server['points'][0]
    requests=[json.loads(s) for s in (root/name/name/'c8-r0-requests.jsonl').read_text().splitlines()]
    telemetry=[json.loads(s) for s in (root/name/name/'c8-r0-telemetry.jsonl').read_text().splitlines()]
    begin=max(r['token_events'][0]['seconds'] for r in requests)
    end=min(r['ended_seconds'] for r in requests)
    count=sum(e['delta_tokens'] for r in requests for e in r['token_events'] if begin<=e['seconds']<=end)
    peak=max(d['used_mib'] for t in telemetry for d in t.get('gpu',{}).get('devices',[]))
    row=dict(name=name,summary=point['summary'],client_eight_active_window=dict(start_seconds=begin,end_seconds=end,output_tokens=count,output_tokens_per_second=count/(end-begin)),peak_sampled_memory_mib=peak,model_throughput_qualified=False)
    rows.append(row)
    print(json.dumps(row,indent=2))
(root/'comparison.json').write_text(json.dumps(dict(schema='tensor.h200-native-comparison.v1',profiles=rows,model_throughput_qualified=False,full_stress_target_reached=False),indent=2)+'\n')
