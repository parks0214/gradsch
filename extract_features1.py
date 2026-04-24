import os
import csv
import hashlib
from scapy.all import *
import glob
from collections import defaultdict
import numpy as np

# 각 플로우(Flow)에서 추출할 패킷의 개수를 20개로 고정
SEQ_LEN = 20

# 폴더명과 라벨(정답) 매핑 딕셔너리
LABEL_MAP = {
    "pcap":0,               # 정상 데이터
    "pcap_keepalive":1,     # keepalive 데이터
    "pcap_stun":2,          # stun 데이터
    "pcap_rdp":3            # rdp 데이터
}

def flow_key(pkt):
    """
    패킷(pkt)에서 5-Tuple (출발지 IP, 출발지 포트, 목적지 IP, 목적지 포트, 프로토콜) 정보를
    추출하여 고유한 플로우(Flow) 키로 반환합니다. 이 키를 통해 같은 세션(응답/요청)의 패킷들을 묶을 수 있습니다.
    """
    # IP 계층이 없는 패킷은 무시
    if IP not in pkt:
        return None

    proto = pkt[IP].proto

    # TCP 패킷인 경우 (프로토콜 번호: 6)
    if proto == 6 and TCP in pkt:
        return (
            pkt[IP].src,
            pkt[TCP].sport,
            pkt[IP].dst,
            pkt[TCP].dport,
            6
        )

    # UDP 패킷인 경우 (프로토콜 번호: 17)
    if proto == 17 and UDP in pkt:
        return (
            pkt[IP].src,
            pkt[UDP].sport,
            pkt[IP].dst,
            pkt[UDP].dport,
            17
        )

    # TCP나 UDP가 아니면 None 반환
    return None

def is_quic_initial(pkt):
    """
    해당 패킷이 QUIC 프로토콜의 Initial 패킷인지 확인합니다.
    (주로 UDP 페이로드의 첫 바이트를 통해 판별)
    """
    # QUIC은 UDP 위에서 동작하므로 UDP가 아니면 제외
    if UDP not in pkt:
        return False

    payload = bytes(pkt[UDP].payload)

    # 페이로드가 비어있으면 제외
    if len(payload) < 1:
        return False

    first = payload[0]

    # QUIC Long Header인지 확인 (첫 바이트의 최상위 비트가 1이어야 함)
    if (first & 0x80) == 0:
        return False

    # 패킷 타입(Packet Type) 추출: 첫 바이트의 상위 2~3번째 비트 (0b00110000)
    pkt_type = (first >> 4) & 0x03

    # Type이 0이면 Initial 패킷
    return pkt_type == 0

def extract_quic_features(pkts):
    """
    제공된 패킷 리스트에서 QUIC 관련 특성(Feature)들을 추출합니다.
    """
    initial_count = 0  # QUIC Initial 패킷 개수
    total_len = 0      # 전체 Initial 패킷의 페이로드 길이 합
    cipher_count = 0   # 특정 바이트(0x13 등) 출현 빈도
    ext_count = 0      # 확장 블록(0x00) 출현 빈도

    for p in pkts:
        # Initial 패킷만 대상
        if not is_quic_initial(p):
            continue

        initial_count += 1
        data = bytes(p[UDP].payload)
        total_len += len(data)

        # 암호화 스위트나 특정 TLS 패턴 출현 횟수를 카운팅
        cipher_count += data.count(b'\x13')
        ext_count += data.count(b'\x00')

    # 추출된 4가지 특성 리스트로 반환
    return [
        initial_count,
        total_len,
        cipher_count,
        ext_count
    ]

def extract_sequence(pkts):
    """
    패킷 시퀀스에서 처음 N개(SEQ_LEN) 패킷의 크기(length)와 
    패킷 간 도착 시간 간격(Inter-Arrival Time, IAT)을 추출합니다.
    """
    lens = []   # 패킷 크기(Payload bytes) 리스트
    iats = []   # 도착 시간 간격 리스트

    base = pkts[0].time  # 첫 패킷 시간 기준점
    prev = base

    for i, p in enumerate(pkts):
        # 지정된 시퀀스 길이를 채우면 중단
        if i >= SEQ_LEN:
            break

        # 페이로드 길이
        plen = len(bytes(p.payload))
        lens.append(plen)

        t = float(p.time)
        iats.append(t - prev)
        prev = t

    # 만약 패킷 수가 SEQ_LEN(20)보다 적으면 0으로 패딩(Zero padding)
    while len(lens) < SEQ_LEN:
        lens.append(0)
        iats.append(0)

    # 크기 배열과 IAT 배열을 합쳐서 반환
    return lens + iats

def flow_stats(pkts):
    """
    하나의 플로우에 대한 전체 통계 정보(지속 시간, 패킷 전송률, 바이트 전송률)를 계산합니다.
    """
    start = float(pkts[0].time)
    end = float(pkts[-1].time)

    # 플로우의 전체 지속 시간 (초 단위)
    dur = end - start

    pkt_count = len(pkts)

    # 해당 플로우 내 모든 페이로드 바이트 수 합계
    bytes_total = sum(len(bytes(p.payload)) for p in pkts)

    # 패킷 전송 속도 (패킷 수 / 시간, 영으로 나누어지는 오류 방지를 위해 0.0001 추가)
    pkt_rate = pkt_count / (dur + 0.0001)

    # 바이트 전송 속도 (바이트 수 / 시간)
    byte_rate = bytes_total / (dur + 0.0001)

    return [
        dur,
        pkt_rate,
        byte_rate
    ]

def ja4q_hash(features):
    """
    추출된 QUIC 특성(features)들을 바탕으로 단순한 형태의 지문(JA4Q와 유사한 해시값)을 생성합니다.
    """
    # 특성들을 쉼표로 연결된 문자열로 변환
    s = ",".join(map(str, features))

    # MD5 해시값을 구한 뒤 앞의 8자리만 16진수 숫자로 변환
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)

def process_pcap(file, label):
    """
    하나의 pcap 파일을 읽어서 전체 플로우를 분리하고, 
    우리가 원하는 피쳐 데이터를 추출하여 리스트(행렬 형태)로 반환합니다.
    """
    try:
        # scapy를 이용해 pcap(패킷 캡처) 파일 읽기
        pkts = rdpcap(file)
    except Exception as e:
        # 지원되지 않는(비어있거나 손상된) 파일 포맷의 경우 오류를 무시하고 통과
        print(f"Error reading {file}: {e}. Skipping.")
        return []

    # 고유 5-Tuple 키를 기준으로 한 세션인지 식별하여 묶기 위함
    flows = defaultdict(list)

    for p in pkts:
        k = flow_key(p)
        if not k:
            continue
        # 같은 플로우 키를 가지는 패킷들을 한 곳에 모음
        flows[k].append(p)

    rows = []

    for fpkts in flows.values():
        # 한 플로우 내에 패킷이 5개 미만이면 의미 없는 통신으로 간주하고 스킵
        if len(fpkts) < 5:
            continue

        # 1. 패킷 순차적 특성 추출 (크기 시퀀스 + 타임 간격)
        seq = extract_sequence(fpkts)

        # 2. 플로우 전체 통계 특성 (지속 시간, 통신 속도 등)
        stats = flow_stats(fpkts)

        # 3. QUIC 초기 패킷 관련 특성 확인 및 추출
        quic = extract_quic_features(fpkts)

        # 4. QUIC 특성 기반 해시(지문) 생성
        ja4 = ja4q_hash(quic)

        # 위에서 추출한 4가지 특성과, 인자로 받은 정답 라벨(label)을 병합해 하나의 배열(행)로 완성
        row = seq + stats + quic + [ja4, label]
        rows.append(row)

    return rows

def main():
    """
    실행 진입점. 
    LABEL_MAP에 정의된 폴더를 순회하며 `.pcap` 파일의 특징을 추출하고,
    그 결과를 `dataset.csv` 파일로 저장합니다.
    """
    rows = []

    # LABEL_MAP을 반복하면서 폴더별로 패킷 추출 작업 수행
    for d, l in LABEL_MAP.items():
        # 해당 폴더(d) 안의 모든 pcap 파일의 경로 목록 조회
        files = glob.glob(f"{d}/*.pcap")

        for f in files:
            print("processing", f)
            # pcap 파일 각각을 처리해서 데이터 목록에 추가
            rows += process_pcap(f, l)

    # --- CSV 파일 헤더(컬럼명) 구성 ---
    
    # 1. 시퀀스 페이로드 길이 + 시퀀스 IAT 헤더 (총 40개: 패킷크기 20개 + 시간간격 20개)
    headers = [f"seq{i}" for i in range(SEQ_LEN * 2)]
    
    # 2. 플로우 시간/속도 헤더
    headers += ["flow_dur", "pkt_rate", "byte_rate"]
    
    # 3. QUIC 통신 관련 헤더
    headers += ["quic_init", "quic_len", "quic_cipher", "quic_ext"]
    
    # 4. 해시 및 정답 라벨 헤더
    headers += ["ja4q", "label"]

    # 추출된 행들을 'dataset.csv' 에 쓰기
    with open("dataset.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)   # 첫 번째 줄에 헤더(컬럼명) 기록
        writer.writerows(rows)     # 두 번째 줄부터 결과 데이터 기록


if __name__ == "__main__":
    main()