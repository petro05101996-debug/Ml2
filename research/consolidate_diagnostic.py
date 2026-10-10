"""Recompute results from immutable answer files; no training or metric tuning."""
from pathlib import Path
import hashlib, html, json, math, shutil

ROOT=Path('collected');OUT=Path('deliverable');OUT.mkdir(exist_ok=True)
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
seeds=[261010,261011];arms=['base','adapter_0.25','adapter_1.0'];results={};baselines={};audits=[]
for seed in seeds:
    d=ROOT/f'qwen-final-diag-{seed}'
    integrity=load(d/'integrity.json')
    for name,expected in integrity.items():
        if sha(d/name)!=expected:raise ValueError('integrity mismatch '+str(d/name))
    report=load(d/'report.json');answers=load(d/'answers.json');cases=load(d/'cases.json');training=load(d/'training.json')
    if report['training_seed']!=seed or len(training)!=48 or [x['step'] for x in training]!=list(range(1,49)):raise ValueError('training mismatch')
    if not report['frozen_tail_unchanged']:raise ValueError('frozen tail changed')
    canonical=lambda x:json.dumps(x['messages'],sort_keys=True,ensure_ascii=False,separators=(',',':'))
    training_prompts={canonical(x) for x in cases if x['split']=='train'}
    test={x['id']:x for x in cases if x['split']=='test'}
    excluded=sorted(k for k,x in test.items() if canonical(x) in training_prompts)
    if excluded!=report['excluded_exact_training_prompt_ids']:raise ValueError('exclusion mismatch')
    metrics={};statuses={};order_instability={};margins={}
    for arm in arms:
        rows=[x for x in answers if x['arm']==arm]
        if len(rows)!=24 or len({x['id'] for x in rows})!=24:raise ValueError('answer coverage')
        grouped={}
        for x in rows:
            if x['id'] not in test or x['label']!=test[x['id']]['label']:raise ValueError('label mismatch')
            if len(x['logits'])!=2 or not all(math.isfinite(v) for v in x['logits']):raise ValueError('invalid logits')
            pred=0 if x['logits'][0]>=x['logits'][1] else 1
            if pred!=x['prediction']:raise ValueError('prediction mismatch')
            if x['id'] in excluded:continue
            grouped.setdefault(x['case_id'],[]).append(x)
        if len(grouped)!=10 or any(len(v)!=2 for v in grouped.values()):raise ValueError('paired coverage')
        status={cid:all(x['prediction']==x['label'] for x in v) for cid,v in grouped.items()};statuses[arm]=status
        unstable=[cid for cid,v in grouped.items() if sum(x['prediction']==x['label'] for x in v)==1]
        order_instability[arm]=sorted(unstable)
        m={'ordered_correct':sum(x['prediction']==x['label'] for v in grouped.values() for x in v),'ordered_total':20,'both_orders_correct':sum(status.values()),'cases':10,'by_family':{f:sum(status[cid] for cid,v in grouped.items() if v[0]['family']==f) for f in sorted({v[0]['family'] for v in grouped.values()})}}
        for k in ['ordered_correct','ordered_total','both_orders_correct','cases','by_family']:
            if m[k]!=report['principal_metrics'][arm][k]:raise ValueError('metric mismatch')
        metrics[arm]=m
        margins[arm]={cid:[{'id':x['id'],'correct_minus_wrong_logit':x['logits'][x['label']]-x['logits'][1-x['label']]} for x in sorted(v,key=lambda x:x['id'])] for cid,v in sorted(grouped.items())}
    for arm in arms[1:]:
        metrics[arm]['gained_cases']=sorted(cid for cid,v in statuses[arm].items() if v and not statuses['base'][cid])
        metrics[arm]['lost_cases']=sorted(cid for cid,v in statuses[arm].items() if not v and statuses['base'][cid])
    baselines[seed]={x['id']:x['logits'] for x in answers if x['arm']=='base'}
    results[str(seed)]={'metrics':metrics,'order_sensitive_cases':order_instability,'correct_option_margins':margins,'excluded_exact_training_prompts':excluded,'identity_logit_max_abs_difference':report['identity_max_logit_difference'],'case_status':statuses}
    audits.append({'seed':seed,'verified_files':len(integrity),'steps':len(training),'raw_answer_records':len(answers),'source_report_sha256':sha(d/'report.json'),'source_answers_sha256':sha(d/'answers.json'),'source_adapter_sha256':sha(d/'adapter-new-diagnostic.safetensors')})
    shutil.copytree(d,OUT/f'seed-{seed}',dirs_exist_ok=True)
base_diff=max(abs(a-b) for k,v in baselines[seeds[0]].items() for a,b in zip(v,baselines[seeds[1]][k]))
summary={'status':'completed','base':'Qwen3-4B-Instruct-2507','revision':'cdbee75f17c01a7cc42f958dc650907174af0554','new_diagnostic_not_old_checkpoint':True,'cache_inputs':48,'training_steps_per_seed':48,'total_training_steps':96,'trainable_parameters_per_adapter':40968,'seeds':seeds,'test_case_count':10,'same_test_cases_for_both_seeds':True,'baseline_max_logit_difference_between_executors':base_diff,'audits':audits,'results':results,'infrastructure':{'execution':'GitHub Actions; real CPU Qwen computation and LoReFT optimization','main_branch_modified':False,'old_private_archives_uploaded':False,'cache_run_id':38036207864,'training_run_id':38036722290,'local_container_status':'ClientError before executing native commands; external executor bypassed this blocker'},'interpretation':'No net improvement in the principal paired decision score. Seed 261011 gains partial-field atomicity and loses signature carry-forward; seed 261010 loses carry-forward without a gain. Reduced intervention changes no case decisions. These are recognition tests, not free generation or professional qualification.'}
write(OUT/'SUMMARY.json',summary)
lines=['# Реальный запуск Qwen и нового LoReFT: итог','', 'Вычислительная блокировка обойдена через подключённый GitHub. Два обучения и проверка завершены. Самостоятельных действий пользователя для запуска не потребовалось.','', '**Это новый небольшой диагностический адаптер на той же базе, а не тест старого личного checkpoint и не завершение всего Round 2.1.**','', '## Измеренные результаты','', '| Вариант | Первый запуск, seed 261010 | Второй запуск, seed 261011 |','|---|---:|---:|']
names={'base':'Без адаптера','adapter_0.25':'Адаптер: сила 0,25','adapter_1.0':'Адаптер: сила 1,0'}
for arm in arms:lines.append('| '+names[arm]+' | '+' | '.join(str(results[str(s)]['metrics'][arm]['both_orders_correct'])+'/10' for s in seeds)+' |')
lines+=['','Зачёт сценария требует правильного выбора при обоих порядках A/B. Это десять сценариев, а не двадцать независимых задач; повтор обучения также не создаёт новую тестовую выборку.','', 'Во втором запуске исправился test-atomicity-1 (независимая обработка полей PATCH), но потерян test-authority_versions-1 (сохранение подписей согласно приоритетной поправке). В первом запуске потерян тот же случай с подписями без приобретения нового. Общего прироста нет.','', 'У базы в пяти из десяти сценариев выбор между содержательными решениями менялся после перестановки A/B. Поэтому исходные 15/20 отдельных правильных выборов не равны устойчивому решению 75% сценариев: оба порядка пройдены только в 5/10.','', '## Что фактически выполнено','', 'Получены 48 настоящих промежуточных состояний Qwen. Обучены два отдельных rank-8 LoReFT по 48 шагов, суммарно 96. У каждого 40 968 параметров. Основная модель не обучалась; сохранённые хеши последнего блока до/после совпали. Проверено тождественное вмешательство. Вычисления теста используют реальные логиты двух токенов A/B. Это не эмуляция Qwen и не ответы другого ассистента.','', '## Ограничения и разделение результатов','', 'Два исходных тестовых сценария external_effect дословно повторяли обучение. Они исключены из главного сравнения по правилу, зафиксированному до просмотра результатов (commit 531b706dd8ff518e0a0a5fd16e1c98fb47be11ce). Все исходные ответы и сырые показатели сохранены.','', 'Оставшиеся сценарии — близкие параметризованные родственники учебных шаблонов. Проверялось различение двух готовых решений на английском, не самостоятельное написание русских требований, выявление всех зависимостей или подготовка законченной спецификации. Перенос профессиональных качеств этим измерением не установлен.','', 'Использован общий для всех вариантов NF4-префикс с FP32 вычислениями и исходный финальный блок в FP32. Нельзя напрямую сравнивать новые проценты со старым опытом в другой численной конфигурации. Токенизатор выдал regex-предупреждение; между вариантами не менялся, причинное влияние предупреждения не проверялось.','', 'Приведены оба заранее заданных seed и обе силы вмешательства без выбора лучшего по тесту. Новое изменение обучения по этим результатам ещё не запускалось.','', '## Что этот результат меняет в дальнейшей работе','', 'Работающий внешний путь загрузки модели, подготовки кэша и настоящего обучения проверен. Не требуется перекладывать настройку на пользователя. Эта небольшая конфигурация адаптера не продвигается как улучшенный аналитик: следующая исследовательская цель — сохранять правильные решения при исправлении ошибок и проверять устойчивость к несущественным изменениям задания. Сокращённый тест остаётся диагностикой, а не заменой комплексной задачи системного аналитика.','', '## Материалы','', 'Подкаталоги seed-261010 и seed-261011 содержат веса, все логиты и решения, учебный журнал, исходные задачи, протокол и код. SUMMARY.json содержит независимо пересчитанные метрики и различия по случаям. MANIFEST.json содержит контрольные суммы комплекта. Большие публичные веса Qwen и кэш состояний не входят в этот малый архив; сохранён путь их воспроизводимого создания в сценариях research и истории отдельной ветки. Основная ветка не изменялась; прежние пользовательские архивы не публиковались.']
md='\n'.join(lines)+'\n';(OUT/'SUMMARY_RU.md').write_text(md)
# Simple readable report; original reports have per-case full texts.
body='<h1>Qwen и LoReFT: реальный диагностический результат</h1><p><strong>Вычисления восстановлены. Два обучения завершены. Общего прироста качества в этом опыте нет.</strong></p><p>Это новый диагностический адаптер, не прежний checkpoint. Проверяется выбор готового решения, не подготовка целой спецификации.</p><table><tr><th>Вариант</th><th>Первый запуск</th><th>Второй запуск</th></tr>'
for arm in arms:body+='<tr><td>'+names[arm]+'</td>'+''.join('<td>'+str(results[str(s)]['metrics'][arm]['both_orders_correct'])+'/10</td>' for s in seeds)+'</tr>'
body+='</table><p>Сценарий засчитывается только при правильном выборе в обоих порядках вариантов. Во втором запуске получен один новый правильный случай с частичным обновлением, но потерян случай с переносом подписей.</p><h2>Полный итог</h2><pre>'+html.escape(md)+'</pre><h2>Исходные отчёты</h2><p><a href="seed-261010/Report_RU.html">Первый запуск</a> · <a href="seed-261011/Report_RU.html">Второй запуск</a></p>'
(OUT/'Report_RU.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Итог реального запуска адаптера</title><style>body{font:16px/1.65 system-ui;max-width:1020px;margin:32px auto;padding:0 22px;color:#17243a}h1{line-height:1.2}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:12px;border-bottom:1px solid #ccd3df}pre{white-space:pre-wrap;font:15px/1.7 system-ui;background:#f3f5f8;padding:18px;border-radius:8px}</style><main>'+body+'</main></html>')
for p in Path('research').glob('*.py'):shutil.copyfile(p,OUT/p.name)
shutil.copyfile('research/diagnostic_pre_evaluation_note.json',OUT/'pre_evaluation_exclusion_note.json')
shutil.copyfile('.github/workflows/qwen-offline-recovery.yml',OUT/'packaging_workflow.yml')
write(OUT/'MANIFEST.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='MANIFEST.json'})
print('CONSOLIDATED_RESULT',json.dumps({k:v for k,v in summary.items() if k!='results'},ensure_ascii=False),flush=True)
print(md,flush=True)
print('BASELINE_MAX_LOGIT_DIFFERENCE',base_diff,flush=True)
