"""Build the full Copernicus-Bench patch-id -> local-file mapping by hashing
S2 pixel content. CB tiles are fetched from the remote zip via range requests
(5128 tiles, ~1.5 GB of compressed reads) -- run in background."""
import urllib.request, struct, zlib, io, json, hashlib, sys, time
import numpy as np, tifffile, glob, os
URL="https://huggingface.co/datasets/wangyi111/Copernicus-Bench/resolve/main/l2_dfc2020_s1s2/dfc2020.zip"
cd_off=8753593934; cd_size=1877861
OUT=os.environ.get("CB_MAP_OUT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cb_patch_map.json"))

def rng(a,b,tries=4):
    for t in range(tries):
        try:
            r=urllib.request.Request(URL,headers={"Range":f"bytes={a}-{b}"})
            return urllib.request.urlopen(r,timeout=180).read()
        except Exception as e:
            if t==tries-1: raise
            time.sleep(3*(t+1))

cd=rng(cd_off,cd_off+cd_size-1)
ents=[]; p=0
while p<len(cd)-4 and cd[p:p+4]==b"PK\x01\x02":
    nlen,elen,clen=struct.unpack("<HHH",cd[p+28:p+34])
    csize,usize=struct.unpack("<II",cd[p+20:p+28])
    lho,=struct.unpack("<I",cd[p+42:p+46])
    name=cd[p+46:p+46+nlen].decode("utf-8","replace")
    extra=cd[p+46+nlen:p+46+nlen+elen]
    if lho==0xFFFFFFFF or csize==0xFFFFFFFF or usize==0xFFFFFFFF:
        q=0
        while q<len(extra)-4:
            hid,hsz=struct.unpack("<HH",extra[q:q+4])
            if hid==0x0001:
                vals=extra[q+4:q+4+hsz]
                nums=[struct.unpack("<Q",vals[k:k+8])[0] for k in range(0,len(vals)-7,8)]
                it=iter(nums)
                if usize==0xFFFFFFFF: usize=next(it)
                if csize==0xFFFFFFFF: csize=next(it)
                if lho==0xFFFFFFFF: lho=next(it)
            q+=4+hsz
    ents.append((name,lho,csize)); p+=46+nlen+elen+clen

s2=[(n,l,c) for n,l,c in ents if "/s2/" in n and n.endswith(".tif")]
print("CB s2 tiles:",len(s2),flush=True)

# hash local files first
root='/home/timz/scratch/ever-changing-modalities/datasets/DFC2020_official/DFC_Public_Dataset'
loc=glob.glob(f'{root}/*/s2_*/*.tif')
lut={}
for f in loc:
    a=tifffile.imread(f)
    lut[hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()]=os.path.basename(f)
print("local hashed:",len(lut),flush=True)

mapping={}; miss=[]
for i,(name,lho,cs) in enumerate(s2):
    blk=rng(lho,lho+cs+300)
    nl,el=struct.unpack("<HH",blk[26:30])
    meth,=struct.unpack("<H",blk[8:10])
    raw=blk[30+nl+el:30+nl+el+cs]
    d=zlib.decompress(raw,-15) if meth==8 else raw
    a=tifffile.imread(io.BytesIO(d))
    h=hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
    b=os.path.basename(name)
    if h in lut: mapping[b]=lut[h]
    else: miss.append(b)
    if (i+1)%250==0:
        print(f"  {i+1}/{len(s2)}  matched={len(mapping)} missed={len(miss)}",flush=True)
        json.dump({"map":mapping,"miss":miss},open(OUT,"w"))
json.dump({"map":mapping,"miss":miss},open(OUT,"w"))
print(f"DONE matched={len(mapping)}/{len(s2)} missed={len(miss)}",flush=True)
