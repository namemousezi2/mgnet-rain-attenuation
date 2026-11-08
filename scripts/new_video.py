import cv2, numpy as np, torch, torch.nn as nn
from common import ROOT, OUT
dev="cuda"; W,H=640,360
ND=ROOT/"data"/"new_clips"/"New rain GT and light medium and heavy"
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
tonp=lambda t:(t.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
def bar(t): 
    b=np.zeros((34,W,3),np.uint8); cv2.putText(b,t,(10,24),0,0.8,(255,255,255),2); return b
for tag,name in (("IMG_1769","heavy"),("IMG_1773","medium"),("IMG_1772","light")):
    c=cv2.VideoCapture(str(ND/(tag+".MOV"))); N=int(c.get(7)); fps=c.get(5) or 30
    s0=int(N*0.4); n=240; c.set(cv2.CAP_PROP_POS_FRAMES,s0-8); fr=[]
    for _ in range(n+16):
        ok,f=c.read()
        if not ok: break
        fr.append(cv2.resize(f,(W,H),interpolation=cv2.INTER_AREA))
    n=len(fr)-16; T=[torch.from_numpy(f.astype(np.float32)/255).permute(2,0,1).to(dev) for f in fr]
    vw=cv2.VideoWriter(str(OUT/("compare_new_"+name+".mp4")),cv2.VideoWriter_fourcc(*"mp4v"),fps,(W*3,H+34))
    for k in range(8,8+n):
        win=torch.stack([T[k+j] for j in range(-8,9,2)])
        a,b,m=T[k],step(win,mg),step(win,m7)
        top=np.hstack([bar("input (rain)"),bar("MG-Net"),bar("mf7")])
        vw.write(np.vstack([top,np.hstack([tonp(a),tonp(b),tonp(m)])]))
    vw.release(); print(name,"ok",flush=True)
print("DONE")
