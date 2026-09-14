import json
def install(e): e.update({'json_encode':lambda v:json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':')),'json_decode':json.loads,'str':str,'int':int,'real':float,'bool':bool})
