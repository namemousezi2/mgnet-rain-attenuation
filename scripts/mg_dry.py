import cv2, numpy as np, json, torch, torch.nn as nn
from common import ROOT, OUT
dev="cuda"; W,H=640,360; D=ROOT/"data"/"new_clips"
class MF(nn.Module):
    def __init__(s,c=64,n=7):
        super().__init__(); L=[nn.Conv2d(18,c,3,padding=1),nn.ReLU(True)]
        for _ in range(n): L+=[nn.Conv2d(c,c,3,padding=1),nn.ReLU(True)]
        L+=[nn.Conv2d(c,3,3,padding=1)]; s.net=nn.Sequential(*L)
    def forward(s,X): return X[:,6:9]+s.net(X)
from mg_net import MG
ck=ROOT/"experiments_v2"/"checkpoints"
mg=MG().to(dev); mg.load_state_dict(torch.load(ck/"student_mg.pth",map_location=dev)["model_state_dict"]); mg.eval()
m7=MF().to(dev); m7.load_state_dict(torch.load(ck/"student_mf7.pth",map_location=dev)["model_state_dict"]); m7.eval()
@torch.no_grad()
def step(win,mm):
    x=win[4]; md=win.median(0).values
    X=torch.cat([win[2],win[3],win[4],win[5],win[6],x-md],0)[None]
    return mm(X)[0].float().clamp(0,1)
lap=torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],dtype=torch.float32,device=dev)[None,None]
def hf(a): return torch.nn.functional.conv2d(a.mean(0,keepdim=True)[None],lap,padding=1).abs().mean().item()
tonp=lambda t:(t.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
res={}; rows=[]
for fn in ("IMG_1417_norain.MOV","CCTV_norain 1.mp4"):
    c=cv2.VideoCapture(str(D/fn)); N=int(c.get(7)); c.set(cv2.CAP_PROP_POS_FRAMES,min(30,max(0,N-80))); fr=[]
    for _ in range(60):
        ok,f=c.read()
        if not ok: break
        fr.append(cv2.resize(f,(W,H),interpolation=cv2.INTER_AREA))
    T=[torch.from_numpy(f.astype(np.float32)/255).permute(2,0,1).to(dev) for f in fr]
    d6=[];d7=[];h0=[];h6=[];h7=[]; ex=None
    for k in range(8,len(T)-8):
        win=torch.stack([T[k+j] for j in range(-8,9,2)]); x=T[k]; g=step(win,mg); m=step(win,m7)
        d6.append((g-x).abs().mean().item()); d7.append((m-x).abs().mean().item()); h0.append(hf(x)); h6.append(hf(g)); h7.append(hf(m))
        if k==20: ex=(x,g,m)
    res[fn]={"mean_abs_change_MG_255":255*np.mean(d6),"mean_abs_change_mf7_255":255*np.mean(d7),"hf_ratio_MG":np.mean(h6)/np.mean(h0),"hf_ratio_mf7":np.mean(h7)/np.mean(h0)}
    print(fn,res[fn],flush=True)
    x,g,m=ex; y0,x0=H//2-45,W//2-80
    cr=lambda t:cv2.resize(tonp(t)[y0:y0+90,x0:x0+160],(480,270),interpolation=cv2.INTER_NEAREST)
    df=cv2.resize(tonp(((g-x).abs()*20).clamp(0,1))[y0:y0+90,x0:x0+160],(480,270),interpolation=cv2.INTER_NEAREST)
    rows.append(np.hstack([cr(x),cr(g),cr(m),df]))
cv2.imwrite(str(OUT/"mg_dry_check.png"),np.vstack(rows))
json.dump(res,open(OUT/"mg_dry_check.json","w"),indent=1); print("DONE")
