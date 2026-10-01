from pathlib import Path
import subprocess, time
from astra.io import read_yaml, write_yaml, file_hash
root=Path(__file__).resolve().parent
pre=read_yaml(root/'preparation-receipt.yaml')
ledger=read_yaml(root/'producer-source-build.yaml')
new=root/'coco'
if file_hash(new)!=ledger['binary_sha256']: raise RuntimeError('Native binary changed')
records=[]
for pair in pre['parity_pairs']:
    outputs={}
    for kind in ('preserved','new'):
        job=pair[kind]
        work=Path(job['work'])
        target=work/'data'/pair['day']/pair['symbol']/'values.parquet'
        if target.exists(): raise RuntimeError('Fresh parity output required: '+str(target))
        cmd=list(job['command'])
        if kind=='new': cmd[0]=str(new)
        start=time.monotonic()
        with (work/'native.log').open('w') as log:
            result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode or not target.is_file(): raise RuntimeError('Parity replay failed; inspect '+str(work/'native.log'))
        outputs[kind]=file_hash(target)
        records.append({'kind':kind,'day':pair['day'],'symbol':pair['symbol'],'output':str(target),'sha256':outputs[kind],'seconds':time.monotonic()-start})
        print('PARITY',kind,pair['symbol'],round(records[-1]['seconds'],2),flush=True)
    original=pair['original_native_sha256']
    passed=outputs['preserved']==outputs['new']==original
    print('BYTE_PARITY',pair['symbol'],passed,'new_vs_preserved',outputs['new']==outputs['preserved'],flush=True)
    write_yaml(root/'baseline-parity.yaml',{'records':records,'pairs_done':sum(r['kind']=='new' for r in records),'latest_original_correspondence':passed})
for job in pre['pilot_jobs']:
    work=Path(job['work'])
    target=work/'data'/job['day']/job['symbol']/'values.parquet'
    if target.exists(): raise RuntimeError('Fresh native output required: '+str(target))
    cmd=list(job['command_after_build']);cmd[0]=str(new)
    start=time.monotonic()
    with (work/'native.log').open('w') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode or not target.is_file(): raise RuntimeError('H10 replay failed; inspect '+str(work/'native.log'))
    records.append({'kind':'h10','day':job['day'],'symbol':job['symbol'],'output':str(target),'sha256':file_hash(target),'seconds':time.monotonic()-start})
    print('H10_NATIVE',job['day'],job['symbol'],round(records[-1]['seconds'],2),flush=True)
    write_yaml(root/'replay-receipt.yaml',{'binary_sha256':ledger['binary_sha256'],'records':records,'completed_h10_cells':sum(r['kind']=='h10' for r in records),'feature_screen_passed':False,'qualification':'Only replay completion; post-replay source/key/phase/support validation pending'})
