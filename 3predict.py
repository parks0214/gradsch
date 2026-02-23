import argparse
import time
import joblib
import torch
import torch.nn as nn
import numpy as np
import json
import os
from scapy.all import sniff, IP, TCP, UDP
from collections import defaultdict
from scapy.all import Raw

# 📌 설정 (학습과 동일해야 함)
SEQUENCE_LENGTH = 20
input_features = 2 # [Length, IAT]
num_classes = 4

# 📌 세션 상태 파일 경로
SESSION_STATE_FILE = "session_state.json"

# 📌 LSTM 모델 클래스 정의
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

sessions = defaultdict(list)
connection_states = {}
persistent_sessions = {}

def load_session_state():
    global persistent_sessions
    if os.path.exists(SESSION_STATE_FILE):
        try:
            with open(SESSION_STATE_FILE, "r") as f:
                persistent_sessions = json.load(f)
            print(f"[INFO] Loaded {len(persistent_sessions)} sessions from state file.")
        except Exception as e:
            print(f"[ERROR] Failed to load session state: {e}")
            persistent_sessions = {}

def save_session_state():
    try:
        with open(SESSION_STATE_FILE, "w") as f:
            json.dump(persistent_sessions, f, indent=4, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"[ERROR] Failed to save session state: {e}")

def check_duration_and_alert(key_str, start_time):
    current_time = time.time()
    duration = current_time - start_time
    
    thresholds = {
        "1시간": 3600,
        "3시간": 3600 * 3,
        "6시간": 3600 * 6,
        "12시간": 3600 * 12,
        "24시간": 3600 * 24,
        "1주일": 3600 * 24 * 7,
        "1달": 3600 * 24 * 30
    }
    
    session_data = persistent_sessions.get(key_str)
    alerted_list = session_data.get("alerted", [])
    
    for label, seconds in thresholds.items():
        if duration >= seconds:
            if label not in alerted_list:
                msg = f"⚠️ [RAT 의심/장기세션] {key_str} 세션이 {label} 이상 유지되고 있습니다! (Duration: {int(duration)}s)"
                print(f"[{time.strftime('%H:%M:%S')}] {msg}")
                with open("prediction_log.csv", "a") as log:
                    log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},ALERT,LONG_SESSION,{label},{key_str}\n")
                
                alerted_list.append(label)
                
    persistent_sessions[key_str]["alerted"] = alerted_list
    persistent_sessions[key_str]["last_seen"] = current_time
    save_session_state()

# [수정] client_ip를 매개변수로 받아 방향성을 부여합니다.
def extract_sequence_from_session(packets, client_ip):
    seq_data = []
    base_time = float(packets[0].time)
    
    for i, pkt in enumerate(packets):
        if i >= SEQUENCE_LENGTH: break
        
        # 1. Payload (방향성 적용)
        payload = len(pkt[Raw].load) if Raw in pkt else 0
        if IP in pkt and pkt[IP].src != client_ip:
            payload = -payload
        
        # 2. IAT
        cur_time = float(pkt.time)
        iat = cur_time - base_time
        base_time = cur_time
        
        seq_data.extend([payload, iat])
        
    # Padding
    current_len = len(seq_data) // 2
    for _ in range(SEQUENCE_LENGTH - current_len):
        seq_data.extend([0, 0.0])
        
    return seq_data

def get_canonical_key(pkt):
    if TCP in pkt:
        sport, dport = pkt[TCP].sport, pkt[TCP].dport
    elif UDP in pkt:
        sport, dport = pkt[UDP].sport, pkt[UDP].dport
    else:
        return None

    src = (pkt[IP].src, sport)
    dst = (pkt[IP].dst, dport)
    return tuple(sorted([src, dst]))

def packet_handler(pkt):
    if IP not in pkt: return
    if TCP not in pkt and UDP not in pkt: return

    c_key = get_canonical_key(pkt)
    if not c_key: return

    is_tcp = TCP in pkt
    
    # Connection State 관리 (방향 및 상태 추적)
    if c_key not in connection_states:
        src_info = (pkt[IP].src, pkt[TCP].sport if is_tcp else pkt[UDP].sport)
        dst_info = (pkt[IP].dst, pkt[TCP].dport if is_tcp else pkt[UDP].dport)
        
        status = "udp"
        if is_tcp:
            if pkt[TCP].flags == 0x02: # SYN only
                status = "3way-ok"
            else:
                status = "anomaly" # SYN Missed
        
        connection_states[c_key] = {
            'client': src_info,
            'server': dst_info,
            'status': status
        }
    else:
        state = connection_states[c_key]
        if is_tcp and state['status'] == "anomaly":
            if pkt[TCP].flags == 0x02:
                state['status'] = "3way-ok"
                state['client'] = (pkt[IP].src, pkt[TCP].sport)
                state['server'] = (pkt[IP].dst, pkt[TCP].dport)
    
    # [NEW] 순수 제어 패킷(SYN, 빈 ACK 등)은 제외하고 실제 페이로드가 있는 패킷만 분석 버퍼에 추가
    if Raw in pkt:
        sessions[c_key].append(pkt)

def main(interface, window_sec=4):
    print(f"[INFO] sniffing on {interface} (LSTM Mode)...")
    
    load_session_state()
    
    device = torch.device("cpu")
    model = LSTMClassifier(input_features, 64, 2, num_classes)
    try:
        model.load_state_dict(torch.load("lstm_model.pth", map_location=device))
        model.eval()
        scaler = joblib.load("scaler.pkl")
        rf_model = joblib.load("rf_model.pkl") 
        print("[INFO] LSTM & Random Forest Models loaded.")
    except Exception as e:
        print(f"[ERROR] Load failed: {e}")
        return

    label_map = {0: "정상", 1: "KEEP-ALIVE", 2: "STUN", 3: "RDP"}
    
    if not os.path.exists("prediction_log.csv"):
        with open("prediction_log.csv", "w") as log:
            log.write("Timestamp,Protocol,Src_IP,Src_Port,Dst_IP,Dst_Port,Detection,Confidence,Model\n")

    while True:
        sniff(iface=interface, prn=packet_handler, timeout=window_sec, store=0)

        for c_key in list(sessions.keys()):
            pkts = sessions[c_key]
            
            if len(pkts) < SEQUENCE_LENGTH:
                continue

            chunk = pkts[:SEQUENCE_LENGTH]
            sessions[c_key] = pkts[SEQUENCE_LENGTH:]

            # [NEW] 시퀀스 추출 전, 방향성 파악을 위해 클라이언트 IP를 먼저 확보합니다.
            state = connection_states.get(c_key)
            if not state: 
                if TCP in chunk[0]:
                    sport, dport = chunk[0][TCP].sport, chunk[0][TCP].dport
                else:
                    sport, dport = chunk[0][UDP].sport, chunk[0][UDP].dport
                connection_states[c_key] = {
                    'client': (chunk[0][IP].src, sport), 
                    'server': (chunk[0][IP].dst, dport), 
                    'status': 'unknown'
                }
                state = connection_states[c_key]

            client_ip, client_port = state['client']
            server_ip, server_port = state['server']
            status = state['status']

            # ---------------------------------------------------------
            # [Step 1] 통계적 특징 추출 & Random Forest
            # ---------------------------------------------------------
            val_payloads = []
            val_iats = []
            base_time = float(chunk[0].time)
            
            for pkt in chunk:
                # 통계량 추출 시에는 길이에 절대값을 사용합니다.
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
            
            stats_features = np.array([p_mean, p_std, p_min, p_max, i_mean, i_std, i_min, i_max, duration, pkt_count]).reshape(1, -1)
            
            rf_pred = rf_model.predict(stats_features)[0]
            rf_probs = rf_model.predict_proba(stats_features)[0]
            rf_conf = np.max(rf_probs)
            
            used_model = "RF"
            if rf_pred == 0 and rf_conf >= 0.99:
                pred = 0
                confidence = rf_conf
            else:
                # ---------------------------------------------------------
                # [Step 2] 시퀀스 특징 추출 & LSTM
                # ---------------------------------------------------------
                used_model = "LSTM"
                # [수정] 추출 함수에 client_ip 전달하여 음수/양수 방향성 추가
                raw_seq = extract_sequence_from_session(chunk, client_ip)
                seq_scaled_flat = scaler.transform([raw_seq]) 
                input_tensor = torch.tensor(seq_scaled_flat.reshape(1, SEQUENCE_LENGTH, input_features), dtype=torch.float32)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    max_prob, pred_idx = torch.max(probs, 1)
                    
                    pred = pred_idx.item()
                    confidence = max_prob.item()

            sport, dport = client_port, server_port
            
            if sport in [53, 5353] or dport in [53, 5353]: pred = 0
            if (sport == 22 or dport == 22) and pred == 3: pred = 0
            
            # [NEW] Keep-Alive (pred=1)은 오직 웹(80, 443)에서만 탐지하도록 강제
            if pred == 1 and not (sport in [80, 443] or dport in [80, 443]):
                pred = 0 
                # print(f"[DEBUG] Blocked non-web Keep-Alive on port {sport}/{dport}")
                
            if sport == 5938 or dport == 5938:
                pred = 4 
                confidence = 1.0 

            label = label_map.get(pred) if pred in label_map else "Unknown"
            proto_tag = "UDP" if status == "udp" else f"TCP/{status.upper()}"
            conf_str = f"({confidence*100:.1f}%)"
            
            if pred != 0:
                if confidence < 0.7:
                     msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🛑 [IGNORED] [{proto_tag}] {client_ip}:{client_port} → {server_ip}:{server_port} \t탐지: {label} {conf_str} - 신뢰도 부족"
                     print(msg)
                     pred = 0 

            if pred != 0:
                label = label_map.get(pred, "Unknown")
                proto_tag = "UDP" if status == "udp" else f"TCP/{status.upper()}"
                
                log_msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🟢 [RECORDED] [{proto_tag}] {client_ip}:{client_port} → {server_ip}:{server_port} \t탐지: {label} ({confidence*100:.1f}%)"
                print(log_msg)
                
                with open("prediction_log.csv", "a") as log:
                    log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{proto_tag},{client_ip},{client_port},{server_ip},{server_port},{label},{confidence:.4f},{used_model}\n")
                
                if confidence >= 0.9:
                    if not os.path.exists("high_confidence_log.csv"):
                        with open("high_confidence_log.csv", "w") as h_log:
                            h_log.write("Timestamp,Protocol,Src_IP,Src_Port,Dst_IP,Dst_Port,Detection,Confidence,Model\n")
                    
                    with open("high_confidence_log.csv", "a") as h_log:
                        h_log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{proto_tag},{client_ip},{client_port},{server_ip},{server_port},{label},{confidence:.4f},{used_model}\n")

                if pred == 1:
                    key_str = f"{client_ip}_{client_port}_{server_ip}_{server_port}"
                    current_ts = time.time()
                    
                    if key_str not in persistent_sessions:
                        persistent_sessions[key_str] = {
                            "start_time": current_ts,
                            "last_seen": current_ts,
                            "protocol": proto_tag,
                            "src_ip": client_ip,
                            "src_port": client_port,
                            "dst_ip": server_ip,
                            "dst_port": server_port,
                            "alerted": []
                        }
                        save_session_state()
                    else:
                        check_duration_and_alert(key_str, persistent_sessions[key_str]["start_time"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface", required=True)
    parser.add_argument("--window", type=int, default=4) 
    args = parser.parse_args()
    main(interface=args.interface, window_sec=args.window)