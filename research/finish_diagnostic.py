"""Resume the fixed diagnostic from immutable public Qwen states.
The first evaluator stopped before any optimizer step: a safe-load enum needed
explicit allowlisting. Two seeds are fixed here before seeing any test metrics.
"""
import argparse, hashlib, html, json, pathlib, shutil
import torch
from transformers.utils.quantization_config import QuantizationMethod
import reft_diagnostic as exp

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    if a.seed not in [261010,261011]:raise ValueError('unplanned seed')
    torch.serialization.add_safe_globals([QuantizationMethod])
    exp.SEED=a.seed
    # Float masks must match the original final block's FP32 computation.
    # This is a lossless dtype promotion, shared by all arms, not a label change.
    promotions=[]
    for f in exp.ROOT.glob('cache-*/*.pt'):
        if f.name=='tail.pt':continue
        sample=torch.load(f,weights_only=True,map_location='cpu')
        def promote(v):
            if torch.is_tensor(v) and v.is_floating_point():return v.float()
            if isinstance(v,tuple):return tuple(promote(x) for x in v)
            return v
        old=exp.digest(f)
        sample['kwargs']={k:promote(v) for k,v in sample['kwargs'].items()}
        torch.save(sample,f)
        promotions.append({'file':str(f),'original_sha256':old,'promoted_sha256':exp.digest(f)})
    exp.train_eval()
    out=exp.ROOT/'result'
    report=json.loads((out/'report.json').read_text());answers=json.loads((out/'answers.json').read_text());cases=json.loads((out/'cases.json').read_text())
    canon=lambda x:json.dumps(x['messages'],ensure_ascii=False,sort_keys=True,separators=(',',':'))
    train={canon(x) for x in cases if x['split']=='train'}
    excluded=sorted(x['id'] for x in cases if x['split']=='test' and canon(x) in train)
    kept=[x for x in answers if x['id'] not in excluded]
    arms=['base','adapter_0.25','adapter_1.0'];principal={};case_status={}
    for arm in arms:
        rows=[x for x in kept if x['arm']==arm];groups={}
        for row in rows:groups.setdefault(row['case_id'],[]).append(row)
        if any(len(v)!=2 for v in groups.values()):raise ValueError('missing answer order')
        status={k:all(v['prediction']==v['label'] for v in vv) for k,vv in groups.items()};case_status[arm]=status
        principal[arm]={'ordered_correct':sum(x['prediction']==x['label'] for x in rows),'ordered_total':len(rows),'both_orders_correct':sum(status.values()),'cases':len(status),'by_family':{family:sum(status[k] for k,v in groups.items() if v[0]['family']==family) for family in sorted(set(x['family'] for x in rows))}}
    for arm in arms[1:]:
        principal[arm]['gained_cases']=sorted(k for k,v in case_status[arm].items() if v and not case_status['base'][k])
        principal[arm]['lost_cases']=sorted(k for k,v in case_status[arm].items() if not v and case_status['base'][k])
    report.update(training_seed=a.seed,cache_seed=261010,principal_metrics=principal,excluded_exact_training_prompt_ids=excluded,pre_outcome_exclusion_note_commit='531b706dd8ff518e0a0a5fd16e1c98fb47be11ce',replication_seeds_fixed_before_metrics=[261010,261011],old_adapter_evaluated=False)
    report['limits']+=['The two external_effect scenarios exactly duplicate training prompts and are excluded from principal metrics under a pre-outcome rule. Raw metrics remain unchanged.','The tokenizer emitted a regex warning; original exported tokenizer was left unchanged for all arms. Historical tokenizer/numeric equivalence is not claimed.','Two seeds on the same test templates are replication of training, not two independent test sets.']
    exp.write(out/'report.json',report);exp.write(out/'cache_dtype_promotions.json',promotions)
    proto=json.loads((out/'protocol.json').read_text());proto['training_seed']=a.seed;proto['cache_seed']=261010;proto['training_replications']=[261010,261011];exp.write(out/'protocol.json',proto)
    detail=[]
    for cid in sorted(case_status['base']):
        task=next(x for x in cases if x['case_id']==cid and x['label']==0)
        detail.append({'case_id':cid,'task_and_options':task['messages'][1]['content'],'correct_choice_order0':'A','both_orders_correct':{arm:case_status[arm][cid] for arm in arms},'answers':[x for x in kept if x['case_id']==cid]})
    exp.write(out/'case_details.json',detail)
    body='<h1>Реальный диагностический прогон Qwen и LoReFT</h1><p><strong>Это новый небольшой адаптер, а не повторная проверка прежних обученных весов.</strong></p>'
    body+=f'<p>Seed {a.seed}. Завершено {report["train_steps"]} шагов. Обучаемых параметров: {report["trainable_parameters"]}. Последний блок базы не изменился: {report["frozen_tail_unchanged"]}.</p>'
    body+='<h2>Выбор между двумя готовыми аналитическими решениями</h2><table><tr><th>Вариант</th><th>Правильные предъявления</th><th>Сценарии, верные в обоих порядках</th></tr>'
    names={'base':'Без адаптера','adapter_0.25':'Адаптер, сила 0,25','adapter_1.0':'Адаптер, сила 1,0'}
    for arm in arms:
        m=principal[arm];body+=f'<tr><td>{names[arm]}</td><td>{m["ordered_correct"]}/{m["ordered_total"]}</td><td>{m["both_orders_correct"]}/{m["cases"]}</td></tr>'
    body+='</table><p>Главная таблица исключает четыре предъявления двух сценариев, которые дословно повторяют обучение. Исключение зафиксировано до получения результатов. Оставшиеся десять сценариев — близкие параметризованные варианты учебных шаблонов, не независимый профессиональный экзамен.</p>'
    body+='<h2>Что проверено и что не доказано</h2><p>Выполнены настоящие вычисления замороженной Qwen3-4B и обучение одного ранга-8 LoReFT. Сравнение использует одни и те же реальные внутренние состояния, последний блок и вероятности токенов A/B. Это проверка различения готовых решений. Она не подтверждает самостоятельное написание требований, приёмки или длинного документа.</p><p>Численный режим: NF4-префикс, вычисления FP32, исходный последний блок FP32. Прямое сравнение процентов со старым пилотом недопустимо. Влияние regex-предупреждения токенизатора отдельно не измерялось; между вариантами токенизатор не менялся.</p>'
    body+='<h2>Все результаты по сценариям</h2>'
    for x in detail:
        body+='<h3>'+html.escape(x['case_id'])+'</h3><pre>'+html.escape(x['task_and_options'])+'</pre><p>'+html.escape(json.dumps(x['both_orders_correct'],ensure_ascii=False))+'</p>'
    body+='<h2>Воспроизводимость</h2><p>В архиве сохранены исходные задания и варианты, все логиты и решения, журнал обучения, веса нового адаптера, код и контрольные суммы. Старые личные архивы на публичный стенд не передавались. Основная ветка репозитория не изменялась.</p>'
    page='<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Qwen: диагностический прогон</title><style>body{font:16px/1.6 system-ui;max-width:1000px;margin:32px auto;padding:0 20px;color:#17243a}h1{line-height:1.2}table{border-collapse:collapse;width:100%}th,td{padding:10px;border-bottom:1px solid #ccd4de;text-align:left}pre{white-space:pre-wrap;background:#f2f5f8;padding:15px;border-radius:8px}h2{margin-top:32px}</style><main>'+body+'</main></html>'
    (out/'Report_RU.html').write_text(page)
    shutil.copyfile(__file__,out/'finish_diagnostic.py');shutil.copyfile(exp.__file__,out/'reft_diagnostic.py')
    exp.write(out/'integrity.json',{p.name:exp.digest(p) for p in out.iterdir() if p.is_file() and p.name!='integrity.json'})
    print('PRINCIPAL_REPORT',json.dumps(report),flush=True)

if __name__=='__main__':main()
