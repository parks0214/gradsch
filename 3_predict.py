import argparse
from scapy.all import *
import torch
import numpy as np
from collections import defaultdict

from extract_features1 import *

import torch.nn as nn

class Model(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(dim,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,4)
        )
    def forward(self,x):
        return self.net(x)

model=Model(48)
model.load_state_dict(torch.load("model.pt"))
model.eval()

flows=defaultdict(list)

def handler(pkt):

    k=flow_key(pkt)

    if not k:
        return

    flows[k].append(pkt)

    if len(flows[k])<20:
        return

    f=flows[k]

    seq=extract_sequence(f)

    stats=flow_stats(f)

    quic=extract_quic_features(f)

    ja4=ja4q_hash(quic)

    feat=np.array(seq+stats+quic+[ja4])

    x=torch.tensor(feat).float().unsqueeze(0)

    pred=model(x).argmax(1).item()

    if pred!=0:
        print(f"🚨 ALERT - Flow: {k} | Prediction: {pred}")
    else:
        print(f"✅ NORMAL - Flow: {k} | Prediction: {pred}")

    flows[k]=[]

parser=argparse.ArgumentParser()
parser.add_argument("--interface")

args=parser.parse_args()

print(f"[*] 인공지능 모델(model.pt) 로딩 완료.")
print(f"[*] 인터페이스 {args.interface} 에서 패킷 캡처를 시작합니다...")
print(f"[*] 각 통신 세션당 20개의 패킷이 모일 때마다 실시간 예측을 출력합니다.\n")

sniff(iface=args.interface,prn=handler,store=False)