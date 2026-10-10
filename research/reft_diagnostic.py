"""Public synthetic diagnostic. No user archives, checkpoint or task data are used.
This is a NEW rank-8 LoReFT calibration, not evaluation of a private old adapter.
Fixed forced-choice continuations test local decisions, not full specifications.
"""
import argparse, gc, hashlib, json, math, os, pathlib, random, time
import torch
from torch import nn
ROOT=pathlib.Path('diagnostic')
REV='cdbee75f17c01a7cc42f958dc650907174af0554'
SEED=261010
SYSTEM='You are a systems analyst. Use only the stated authoritative rules. Select the supported conclusion. Reply with exactly A or B.'

def write(p,obj):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2))
def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8388608),b''): h.update(b)
    return h.hexdigest()
def cases():
    rows=[]
    def add(split,fam,i,condition,good,bad):
        for reverse in [False,True]:
            choices=[good,bad] if not reverse else [bad,good]
            prompt=condition+'\nA. '+choices[0]+'\nB. '+choices[1]+'\nAnswer:'
            rows.append(dict(id=f'{split}-{fam}-{i}-order{int(reverse)}',case_id=f'{split}-{fam}-{i}',split=split,family=fam,messages=[{'role':'system','content':SYSTEM},{'role':'user','content':prompt}],label=int(reverse)))
    for split in ['train','test']:
        a,b,c=(9,4,7) if split=='train' else (13,6,8)
        for i,threshold in enumerate([b+c,b+c+1]):
            total=b+c;cut=total>=threshold
            cond=f'Authoritative rule: for each business ID only the highest revision contributes. A newer revision replaces, not adds to, the previous amount. Repeated event IDs and lower revisions have no effect. Process the entire batch, then check a single total threshold of {threshold}. If reached, create one job for the entire total and reset the accumulator; otherwise keep the total. Batch: e1 X rev1 amount{a}; exact repeat e1; e2 Y rev1 amount{b}; e3 X rev2 amount{c}; e4 X rev1 amount{a+1}.'
            good=f'Total={total}; job='+str(total if cut else None)+'; remaining='+str(0 if cut else total)+'.'
            bad=f'Total={a+b+c}; job={a+b+c}; remaining=0.'
            add(split,'revision',i,cond,good,bad)
        version=4 if split=='train' else 8
        for i,carry in enumerate([False,True]):
            cond=f'Current document version is {version}, approved with valid signatures A and B. The signed amendment has priority over the old policy and the UI mockup. Amendment: changing content increases version by one; '+('both signatures carry forward and remain valid.' if carry else 'both signatures are invalidated; new signatures on the current version are required.')+f' The content is changed once. A new A signature for version {version+1} then arrives. An additional B signature for old version {version} arrives last. Signatures on older versions are ignored. The non-authoritative UI mockup shows a third signer C.'
            good=f'Current version={version+1}; A valid; B '+('valid; approved.' if carry else 'invalid; awaiting B.')+' C is not required.'
            bad=f'Current version={version+1}; A valid; B '+('invalid; awaiting B.' if carry else 'valid; approved.')+' C is not required.'
            add(split,'authority_versions',i,cond,good,bad)
        v=3 if split=='train' else 7
        for i,atomic in enumerate([True,False]):
            cond=f'Initial profile: name=old, limit=5, version={v}. PATCH requests name=new and limit=null. Authoritative validation forbids null for limit. Processing rule: '+('the whole PATCH is atomic; any invalid field rolls back all changes and the version.' if atomic else 'fields are processed independently; valid fields are saved, invalid fields rejected, and the version increases once if any field is saved.')+' There are no concurrent updates.'
            good=f'name='+('old' if atomic else 'new')+f', limit=5, version={v if atomic else v+1}.'
            bad=f'name='+('new' if atomic else 'old')+f', limit=5, version={v+1 if atomic else v}.'
            add(split,'atomicity',i,cond,good,bad)
        for i,dedupe in enumerate([False,True]):
            cond='A client submitted an operation with key K. It received no response before its timeout. No independent observation of the first attempt exists. It will retry with the SAME key K. The recipient '+('atomically records each key together with its effect; every replay returns the recorded result without repeating the effect.' if dedupe else 'has no deduplication or status-query mechanism; retrying K can create another effect.')+' Network timeout alone does not describe the recipient state.'
            good='The first outcome is unknown. '+('The retry cannot create a second effect for K.' if dedupe else 'A duplicate effect is possible, not certain.')
            bad='The first outcome is unknown. '+('A duplicate effect remains possible despite atomic key/effect deduplication.' if dedupe else 'A retry with the same key cannot duplicate the effect even without recipient deduplication.')
            add(split,'external_effect',i,cond,good,bad)
        for i,external in enumerate([False,True]):
            gen=4 if split=='train' else 11
            cond=f'Worker W1 held generation {gen}; its lease expired. W2 now holds generation {gen+1} and the local task is RUNNING. A late success ACK from W1 must be ignored locally unless its generation equals the current generation. W1 may still attempt an external write. The external resource '+(f'atomically enforces fencing and has already accepted generation {gen+1}; lower generations are rejected.' if external else 'does not check generations or leases.')+' No current W2 result is available.'
            good='Ignore the old ACK; keep local RUNNING. '+('The old external write is rejected by fencing.' if external else 'The old external write is still possible.')
            bad='Ignore the old ACK; keep local RUNNING. '+('The old external write remains possible even with enforced fencing.' if external else 'The old external write is impossible just because its local lease expired.')
            add(split,'fencing',i,cond,good,bad)
        n=0 if split=='train' else 9
        for i,authoritative in enumerate([False,True]):
            cond=f'A proposed UI mockup mentions retry_limit={n}. '+(f'The approved contract explicitly adopts retry_limit={n}; it takes precedence over the mockup.' if authoritative else 'The approved contract states that the retry limit is not yet decided; mockups cannot create a binding policy.')+' Required output distinguishes an unknown limit from an explicitly configured zero or positive limit. There is no default.'
            good='The binding retry limit is '+(str(n)+'.' if authoritative else 'unknown; do not invent a number.')
            bad='The binding retry limit is '+('unknown despite the explicit approved setting.' if authoritative else str(n)+' because the mockup mentions it.')
            add(split,'uncertainty',i,cond,good,bad)
    return rows

def setup():
    torch.set_num_threads(4);torch.manual_seed(SEED)
    import transformers.utils.import_utils as u
    u._torchvision_available=False;u._librosa_available=False
    from transformers.integrations import sdpa_attention as s
    if hasattr(s,'use_gqa_in_sdpa'):
        old=s.use_gqa_in_sdpa
        s.use_gqa_in_sdpa=lambda *a,**k: False

def load_model():
    setup()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from safetensors.torch import load_file
    base=pathlib.Path('portable-qwen')
    manifest=json.loads((base/'portable_manifest.json').read_text())
    if manifest['revision']!=REV:raise ValueError('wrong public base revision')
    # Artifacts already have transport checksums; verify model tensors once per job.
    for name,h in manifest['exported_sha256'].items():
        if digest(base/name)!=h:raise ValueError('bad exported file '+name)
    m=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,trust_remote_code=False,device_map={'':'cpu'},dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True)
    import bitsandbytes as bnb
    for layer in m.modules():
        if isinstance(layer,bnb.nn.Linear4bit):
            layer.compute_dtype=torch.float32;layer.compute_type_is_set=True
            layer.support_avx512bf16_for_cpu=False
    # Preserve the original (not dequantized) public final block, computed in FP32.
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
    last=Qwen3DecoderLayer(m.config,35).float()
    state=load_file(str(base/'original-bf16-suffix.safetensors'))
    state={k.removeprefix('model.layers.35.'):v.float() for k,v in state.items()}
    last.load_state_dict(state,strict=True)
    m.model.layers[35]=last;del state
    m.eval().requires_grad_(False)
    t=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
    return m,t,manifest

class Adapter(nn.Module):
    def __init__(self,width=2560,rank=8):
        super().__init__();g=torch.Generator().manual_seed(SEED)
        r=torch.linalg.qr(torch.randn(width,rank,generator=g),mode='reduced').Q.contiguous()
        self.basis=nn.Parameter(r);self.project=nn.Parameter(r.T.clone().contiguous());self.bias=nn.Parameter(torch.zeros(rank))
    def delta(self,h):
        r=torch.linalg.qr(self.basis,mode='reduced').Q.T
        return (h@self.project.T+self.bias-h@r.T)@r
    def forward(self,h,alpha=1.0):return h+alpha*self.delta(h)

class Capture(Exception):pass

def cache_job(shard):
    m,t,manifest=load_model();rows=cases();out=ROOT/f'cache-{shard}';out.mkdir(parents=True,exist_ok=True)
    ids=[t.encode(s,add_special_tokens=False) for s in ['A','B']]
    if any(len(x)!=1 for x in ids):raise ValueError('labels not single tokens')
    label_ids=[x[0] for x in ids]
    # Tail and two exact vocabulary rows are enough for forced-choice evaluation.
    if shard==0:
        torch.save({'layer':m.model.layers[35].state_dict(),'norm':m.model.norm.weight.detach().float().clone(),'head':m.lm_head.weight.detach()[label_ids].float().clone(),'config':m.config.to_dict(),'label_ids':label_ids,'manifest':manifest},out/'tail.pt')
        write(out/'protocol.json',{'seed':SEED,'revision':REV,'training_steps':48,'learning_rate':0.0001,'relative_delta_penalty':0.01,'alpha_arms':[0,0.25,1.0],'measure':'two-label probability at fixed prompt; no free-generation claim','training_cases':12,'training_ordered_inputs':24,'test_cases':12,'test_ordered_inputs':24,'numerics':'NF4 frozen prefix with FP32 matrix computation and original FP32 final block; matched arms','cases_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()})
        write(out/'cases.json',rows)
    selected=[r for i,r in enumerate(rows) if i%6==shard]
    for row in selected:
        start=time.monotonic();batch=t.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True)
        captured={}
        def hook(module,args,kwargs):
            captured['hidden']=args[0].detach().float().clone()
            def cp(x):
                if torch.is_tensor(x):return x.detach().clone()
                if isinstance(x,tuple):return tuple(cp(v) for v in x)
                return x
            captured['kwargs']={k:cp(v) for k,v in kwargs.items() if k not in ['past_key_value','past_key_values','use_cache']}
            raise Capture()
        handle=m.model.layers[35].register_forward_pre_hook(hook,with_kwargs=True)
        try:
            with torch.inference_mode():m.model(**batch,use_cache=False)
        except Capture:pass
        finally:handle.remove()
        if not captured:raise ValueError('capture not reached')
        captured.update({k:row[k] for k in ['id','case_id','family','split','label']})
        captured['input_tokens']=batch['input_ids'].numel();captured['seconds']=time.monotonic()-start
        torch.save(captured,out/(row['id']+'.pt'))
        print('CACHE',row['id'],captured['input_tokens'],round(captured['seconds'],2),flush=True)
    write(out/'complete.json',{'shard':shard,'count':len(selected),'complete':True})

def train_eval():
    setup()
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
    files=list(ROOT.glob('cache-*/*.pt'));tailfile=next(p for p in files if p.name=='tail.pt')
    z=torch.load(tailfile,weights_only=True,map_location='cpu')
    cfg=Qwen3Config.from_dict(z['config']);cfg._attn_implementation='sdpa'
    layer=Qwen3DecoderLayer(cfg,35).float();layer.load_state_dict(z['layer']);layer.eval().requires_grad_(False)
    norm=z['norm'];head=z['head'];eps=cfg.rms_norm_eps
    adapter=Adapter();initial={k:v.clone() for k,v in adapter.state_dict().items()}
    samples=[torch.load(p,weights_only=True,map_location='cpu') for p in files if p.name!='tail.pt']
    if len(samples)!=48 or len(set(x['id'] for x in samples))!=48:raise ValueError('cache incomplete')
    training=sorted([x for x in samples if x['split']=='train'],key=lambda x:x['id'])
    testing=sorted([x for x in samples if x['split']=='test'],key=lambda x:x['id'])
    def logits(sample,alpha=0):
        h=sample['hidden'].clone();d=adapter.delta(h[:,-1:])
        if alpha:h=torch.cat([h[:,:-1],h[:,-1:]+alpha*d],dim=1)
        kw=sample['kwargs']
        y=layer(h,**kw)
        if isinstance(y,tuple):y=y[0]
        y=y[:,-1].float();y=y*torch.rsqrt(y.square().mean(-1,keepdim=True)+eps)*norm
        return y@head.T,d.square().mean()/sample['hidden'][:,-1:].square().mean().clamp_min(1e-8)
    baseline=[]
    with torch.no_grad():
        for x in testing:
            l,_=logits(x,0);baseline.append({'id':x['id'],'case_id':x['case_id'],'family':x['family'],'label':x['label'],'arm':'base','logits':l[0].tolist(),'prediction':int(l.argmax(-1))})
        # At exact identity, adapter must agree numerically with disabled computation.
        l0,_=logits(training[0],0);l1,_=logits(training[0],1)
        identity_diff=float((l0-l1).abs().max())
        if identity_diff>1e-3:raise ValueError('identity parity failure')
    # Baseline recorded, but never used to choose train samples or hyperparameters.
    out=ROOT/'result';out.mkdir(parents=True,exist_ok=True)
    write(out/'baseline.json',baseline)
    opt=torch.optim.AdamW(adapter.parameters(),lr=0.0001,weight_decay=0.0)
    history=[];rng=random.Random(SEED)
    frozen_before={k:hashlib.sha256(v.detach().numpy().tobytes()).hexdigest() for k,v in layer.state_dict().items()}
    for epoch in range(2):
        order=list(training);rng.shuffle(order)
        for x in order:
            opt.zero_grad(set_to_none=True);l,pen=logits(x,1)
            ce=torch.nn.functional.cross_entropy(l,torch.tensor([x['label']]))
            loss=ce+0.01*pen
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.0);opt.step()
            if any(p.grad is not None for p in layer.parameters()):raise ValueError('base received gradients')
            history.append({'step':len(history)+1,'id':x['id'],'loss':float(loss.detach()),'ce':float(ce.detach()),'relative_delta_squared':float(pen.detach())})
            print('STEP',history[-1],flush=True)
    frozen_after={k:hashlib.sha256(v.detach().numpy().tobytes()).hexdigest() for k,v in layer.state_dict().items()}
    if frozen_before!=frozen_after:raise ValueError('frozen tail changed')
    from safetensors.torch import save_file
    save_file(adapter.state_dict(),str(out/'adapter-new-diagnostic.safetensors'))
    write(out/'training.json',history)
    answers=baseline
    with torch.no_grad():
        for alpha in [0.25,1.0]:
            for x in testing:
                l,pen=logits(x,alpha);answers.append({'id':x['id'],'case_id':x['case_id'],'family':x['family'],'label':x['label'],'arm':f'adapter_{alpha}','logits':l[0].tolist(),'prediction':int(l.argmax(-1)),'relative_delta_squared':float(pen)})
    write(out/'answers.json',answers)
    metrics={}
    for arm in ['base','adapter_0.25','adapter_1.0']:
        r=[x for x in answers if x['arm']==arm];bycase={}
        for x in r:bycase.setdefault(x['case_id'],[]).append(x['prediction']==x['label'])
        metrics[arm]={'ordered_correct':sum(x['prediction']==x['label'] for x in r),'ordered_total':len(r),'both_orders_correct':sum(all(v) for v in bycase.values()),'cases':len(bycase),'by_family':{f:sum(all(v) for k,v in bycase.items() if k.startswith('test-'+f+'-')) for f in sorted(set(x['family'] for x in r))}}
    report={'status':'completed','new_experiment':True,'old_adapter_evaluated':False,'train_steps':len(history),'trainable_parameters':sum(p.numel() for p in adapter.parameters()),'frozen_tail_unchanged':frozen_before==frozen_after,'identity_max_logit_difference':identity_diff,'metrics':metrics,'limits':['12 synthetic test scenarios, each shown in both answer orders; not 24 independent tasks','Forced binary choice using exact token logits; not free generation or finished requirements','New diagnostic adapter trained on public synthetic data, not the previously trained private adapter','One fixed training seed; no checkpoint or alpha selected after seeing test results','All arms share the same cached Qwen prefix and original FP32 final block; numeric configuration differs from historical pilot']}
    write(out/'report.json',report);print('FINAL_REPORT',json.dumps(report),flush=True)
    for src in [ROOT/'cache-0'/'protocol.json',ROOT/'cache-0'/'cases.json']:
        (out/src.name).write_bytes(src.read_bytes())
    write(out/'integrity.json',{p.name:digest(p) for p in out.iterdir() if p.is_file()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['cache','train_eval']);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
    cache_job(a.shard) if a.mode=='cache' else train_eval()
