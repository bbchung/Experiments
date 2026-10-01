from pathlib import Path
import subprocess,time,json
from astra.io import read_yaml,write_yaml,file_hash,digest
root=Path(__file__).resolve().parent
plan=read_yaml(root/'producer-reproduction-plan-v2.yaml')
identity=plan['identity'];body={k:v for k,v in plan.items() if k!='identity'}
if digest(body)!=identity or (root/'producer-reproduction-plan-v2.yaml.identity').read_text().strip()!=identity: raise RuntimeError('Original producer plan changed')
if file_hash(Path(plan['original_binary']['path']))!=plan['original_binary']['sha256']: raise RuntimeError('Original producer bytes changed')
runtime=read_yaml(Path(plan['runtime_snapshot']['path']))
for dep in runtime['dependencies']:
    if file_hash(Path(dep['path']))!=dep['sha256']: raise RuntimeError('Snapshot runtime drift')
import os
native_env={**os.environ,'LD_LIBRARY_PATH':str(root/'runtime')}
results=[]
for job in plan['jobs']:
    work=Path(job['work']);output=Path(job['output'])
    if output.exists(): raise RuntimeError('Fresh original reproduction path required')
    if file_hash(work/'config.yaml')!=job['config_sha256']: raise RuntimeError('Original config drift')
    start=time.monotonic()
    with (work/'native.log').open('w') as log: result=subprocess.run(job['command'],stdout=log,stderr=subprocess.STDOUT,env=native_env)
    if result.returncode or not output.is_file(): raise RuntimeError('Original producer replay failed; '+str(work/'native.log'))
    observed=file_hash(output)
    record={**job,'sha256':observed,'parent_reproduced':observed==job['original']['sha256'],'seconds':time.monotonic()-start}
    results.append(record)
    write_yaml(root/'original-reproduction.yaml',{'schema':'original-producer-reproduction-v1','plan_identity':identity,'records':results,'complete':len(results)==len(plan['jobs']),'passed':len(results)==len(plan['jobs']) and all(r['parent_reproduced'] for r in results)})
    print('ORIGINAL_REPRODUCED',job['symbol'],record['parent_reproduced'],round(record['seconds'],2),flush=True)
