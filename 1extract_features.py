import os
import glob
import csv
import logging
from datetime import datetime
from scapy.all import rdpcap, TCP, UDP, IP, Raw
from collections import defaultdict
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# 라벨 매핑
DEFAULT_LABEL_DIR_MAP = {
    "pcap": 0,          # 일반
    "pcap_keepalive": 1, 
    "pcap_stun": 2,
    "pcap_rdp": 3,
}

SEQUENCE_LENGTH = 20 

# [수정] client_ip 파라미터를 추가하여 방향성을 판별합니다.
def extract_sequence(packets, client_ip):
    seq_data = []
    base_time = None
    if packets and hasattr(packets[0], 'time'):
        base_time = float(packets[0].time)

    for i, pkt in enumerate(packets):
        if i >= SEQUENCE_LENGTH:
            break
            
        # 1. Payload Size (방향성 부여)
        payload_len = 0
        if Raw in pkt:
            payload_len = len(pkt[Raw].load)
            
        # [NEW] Client가 보낸 패킷은 양수(+), Server가 보낸 패킷은 음수(-)로 변환
        if IP in pkt and pkt[IP].src != client_ip:
            payload_len = -payload_len
        
        # 2. IAT (첫 패킷은 0)
        iat = 0.0
        if hasattr(pkt, 'time') and base_time is not None:
             cur_time = float(pkt.time)
             iat = cur_time - base_time
             base_time = cur_time 
        
        seq_data.append([payload_len, iat])

    # 패딩 (20개보다 적으면 0으로 채움)
    while len(seq_data) < SEQUENCE_LENGTH:
        seq_data.append([0, 0.0])
        
    return seq_data

def get_session_key(pkt):
    if TCP in pkt:
        sport, dport = pkt[TCP].sport, pkt[TCP].dport
    elif UDP in pkt:
        sport, dport = pkt[UDP].sport, pkt[UDP].dport
    else:
        return None

    src = (pkt[IP].src, sport)
    dst = (pkt[IP].dst, dport)
    return tuple(sorted([src, dst]))

def process_pcap(file_path, label, min_packets=3):
    try:
        packets = rdpcap(file_path)
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return []

    sessions = defaultdict(list)
    for pkt in packets:
        if IP in pkt and (TCP in pkt or UDP in pkt):
            try:
                key = get_session_key(pkt)
                if key:
                    sessions[key].append(pkt)
            except Exception:
                continue

    data_rows = []
    STRIDE = SEQUENCE_LENGTH 

    for sess_pkts in sessions.values():
        if len(sess_pkts) < min_packets:
            continue
            
        # [NEW] 세션의 방향성 기준이 될 Client IP 식별 (세션의 첫 번째 패킷 전송자)
        client_ip = sess_pkts[0][IP].src
        
        # [NEW] 순수 제어 패킷(SYN, ACK 등) 필터링: Raw 레이어(Payload)가 있는 패킷만 추출
        valid_pkts = [pkt for pkt in sess_pkts if Raw in pkt]
        
        if len(valid_pkts) < min_packets: 
            continue

        num_packets = len(valid_pkts)
        
        # [수정] 노이즈가 제거된 valid_pkts를 대상으로 슬라이딩 윈도우 적용
        for i in range(0, num_packets, STRIDE):
            chunk = valid_pkts[i : i + SEQUENCE_LENGTH]
            
            if len(chunk) < min_packets: 
                continue

            # 방향성이 포함된 시퀀스 추출
            seq = extract_sequence(chunk, client_ip)
            
            # --- 통계적 특징 추출 (for Random Forest) ---
            val_payloads = []
            val_iats = []
            
            base_time = float(chunk[0].time)
            for pkt in chunk:
                # 통계량 추출 시에는 크기 자체의 분포를 보기 위해 절대값을 사용
                p_len = len(pkt[Raw].load) if Raw in pkt else 0
                val_payloads.append(p_len)
                
                cur_time = float(pkt.time)
                iat = cur_time - base_time
                val_iats.append(iat)
                base_time = cur_time 
            
            p_mean = np.mean(val_payloads) if val_payloads else 0
            p_std = np.std(val_payloads) if val_payloads else 0
            p_min = np.min(val_payloads) if val_payloads else 0
            p_max = np.max(val_payloads) if val_payloads else 0
            
            i_mean = np.mean(val_iats) if val_iats else 0
            i_std = np.std(val_iats) if val_iats else 0
            i_min = np.min(val_iats) if val_iats else 0
            i_max = np.max(val_iats) if val_iats else 0
            
            duration = sum(val_iats)
            pkt_count = len(chunk)
            
            stats = [p_mean, p_std, p_min, p_max, i_mean, i_std, i_min, i_max, duration, pkt_count]
            # ---------------------------------------------------

            flat_row = []
            for item in seq:
                flat_row.extend(item)
            
            flat_row.extend(stats)
            flat_row.append(label)
            data_rows.append(flat_row)

    return data_rows

def main():
    all_data = []
    
    headers = []
    for i in range(SEQUENCE_LENGTH):
        headers.append(f"pkt_{i}_len")
        headers.append(f"pkt_{i}_iat")
    
    headers.extend(["p_mean", "p_std", "p_min", "p_max", "i_mean", "i_std", "i_min", "i_max", "duration", "pkt_count"])
    headers.append("label")

    for dirname, label in DEFAULT_LABEL_DIR_MAP.items():
        abs_dir = os.path.abspath(dirname)
        pcap_files = glob.glob(os.path.join(abs_dir, "*.pcap"))
        
        for path in pcap_files:
            logging.info(f"Processing {path} (label={label})")
            rows = process_pcap(path, label)
            all_data.extend(rows)

    if not all_data:
        logging.warning("No data extracted. Check your PCAP paths and contents.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"seq_features_{timestamp}.csv"
    
    with open(out_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(all_data)
        
    logging.info(f"Sequence features saved to {out_filename} (Columns: {len(headers)})")

if __name__ == "__main__":
    main()