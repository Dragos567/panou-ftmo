# Sonda MetaApi: verifica simboluri, preturi live, viteza si adancimea istoricului FTMO
import os, json, time, datetime as dt, urllib.request, urllib.error, urllib.parse
TOKEN=os.environ['METAAPI_TOKEN']; AID=os.environ['METAAPI_ACCOUNT_ID']
PROV=os.environ.get('MA_PROV','https://mt-provisioning-api-v1.agiliumtrade.agiliumtrade.ai')
CLIENT=os.environ.get('MA_CLIENT','https://mt-client-api-v1.{region}.agiliumtrade.ai')
MDATA=os.environ.get('MA_MDATA','https://mt-market-data-client-api-v1.{region}.agiliumtrade.ai')
OUT=os.environ.get('PROBE_OUT','/opt/panou/report.json')
CANDS={
 'Nikkei':['JP225.cash','JP225','JPN225','NIKKEI225','JP225.'],
 'Gold':['XAUUSD','GOLD'],
 'EURUSD':['EURUSD'],
 'USDJPY':['USDJPY'],
 'DAX':['GER40.cash','GER40','DE40.cash','DE40','DAX40','GER30.cash'],
 'GBPUSD':['GBPUSD'],
 'UK100':['UK100.cash','UK100','FTSE100'],
}
TFS=['1m','15m','1h','1d']
R={'started':time.time(),'stage':'start','errors':[]}
def save(stage=None):
    if stage: R['stage']=stage
    R['updated']=time.time()
    tmp=OUT+'.tmp'; open(tmp,'w').write(json.dumps(R,indent=1,default=str)); os.replace(tmp,OUT)
def http(url,tries=3,timeout=25):
    last=None
    for i in range(tries):
        try:
            rq=urllib.request.Request(url,headers={'auth-token':TOKEN,'Accept':'application/json'})
            with urllib.request.urlopen(rq,timeout=timeout) as r:
                b=r.read()
                if r.status==202: last='202'; time.sleep(3); continue
                return json.loads(b) if b else None
        except urllib.error.HTTPError as e:
            body=e.read().decode('utf8','ignore')[:300]; last='HTTP %s %s'%(e.code,body)
            if e.code==404: raise RuntimeError(last)
            if e.code==429:
                try: wait=max(1,(dt.datetime.fromisoformat(json.loads(body)['metadata']['recommendedRetryTime'].replace('Z','+00:00'))-dt.datetime.now(dt.timezone.utc)).total_seconds())
                except Exception: wait=5
                time.sleep(min(wait,30)); continue
            time.sleep(min(2+i*2,15))
        except Exception as e:
            last=repr(e); time.sleep(min(2+i*2,15))
    R.setdefault('log',[]).append('%s -> %s'%(url.split('/users/current/')[-1][:90],last))
    raise RuntimeError(last)
def iso(t): return t.strftime('%Y-%m-%dT%H:%M:%S.000Z')
def candles(base,sym,tf,start=None,limit=1000):
    u='%s/users/current/accounts/%s/historical-market-data/symbols/%s/timeframes/%s/candles?limit=%d'%(base,AID,urllib.parse.quote(sym,safe=''),tf,limit)
    if start: u+='&startTime='+urllib.parse.quote(iso(start))
    return http(u) or []
def depth(base,sym,tf,now):
    lo=dt.datetime(2008,1,1); hi=now
    if not candles(base,sym,tf,hi,1): return None
    for _ in range(9):   # bisect pe luni: exista lumanari inainte de T?
        mid=lo+(hi-lo)/2
        if candles(base,sym,tf,mid,1): hi=mid
        else: lo=mid
    c=candles(base,sym,tf,hi,1000)
    first=min((x['time'] for x in c),default=None)
    return {'earliest_approx':str(first or hi)[:10],'years':round((now-dt.datetime.strptime((first or iso(hi))[:10],'%Y-%m-%d')).days/365.25,2)}
def main():
    save('cont')
    a=http('%s/users/current/accounts/%s'%(PROV,AID))
    region=a.get('region') or 'london'
    R['account']={k:a.get(k) for k in ('name','type','server','region','state','connectionStatus','reliability','platform')}
    base=CLIENT.format(region=region); mbase=MDATA.format(region=region)
    save('simboluri')
    syms=http('%s/users/current/accounts/%s/symbols'%(base,AID))
    R['n_symbols']=len(syms); R['sample_symbols']=syms[:60]
    mp={}
    for k,cs in CANDS.items():
        s=next((c for c in cs if c in syms),None)
        if not s:
            key=k.upper().replace('NIKKEI','JP').replace('DAX','GER')[:3]
            s=next((x for x in syms if x.upper().startswith(key)),None)
        mp[k]=s
    R['symbol_map']=mp
    all_idx=[x for x in syms if x.upper().startswith(('JP','GER','DE','UK','FTSE','NIKK'))]
    R['index_like_symbols']=all_idx
    save('preturi')
    R['specs']={};R['prices']={}
    for k,s in mp.items():
        if not s: continue
        try: R['specs'][k]=http('%s/users/current/accounts/%s/symbols/%s/specification'%(base,AID,urllib.parse.quote(s,safe='')))
        except Exception as e: R['errors'].append('spec %s: %s'%(k,e))
        pr=[]
        for i in range(3):
            try:
                p=http('%s/users/current/accounts/%s/symbols/%s/current-price'%(base,AID,urllib.parse.quote(s,safe='')))
                pr.append({'bid':p.get('bid'),'ask':p.get('ask'),'time':p.get('time'),'brokerTime':p.get('brokerTime')})
            except Exception as e: R['errors'].append('price %s: %s'%(k,e))
            time.sleep(1.2)
        R['prices'][k]=pr
        save()
    save('viteza')
    s0=next(v for v in mp.values() if v)
    t0=time.time(); c=candles(mbase,s0,'1m',None,1000); dtm=time.time()-t0
    R['speed']={'symbol':s0,'tf':'1m','candles':len(c),'seconds':round(dtm,2),
      'first':c[0]['time'] if c else None,'last':c[-1]['time'] if c else None,'sample':c[-2:] }
    save('istoric')
    now=dt.datetime.utcnow().replace(microsecond=0)
    R['history']={}
    for k,s in mp.items():
        if not s: continue
        R['history'][k]={}
        for tf in TFS:
            try: R['history'][k][tf]=depth(mbase,s,tf,now)
            except Exception as e: R['history'][k][tf]='eroare: %s'%e
            save()
    R['finished']=time.time(); save('gata')
if __name__=='__main__':
    try: main()
    except Exception as e:
        R['errors'].append('FATAL %r'%e); save('eroare')
