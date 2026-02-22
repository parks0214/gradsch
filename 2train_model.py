import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import glob
import os
import matplotlib.pyplot as plt

# 📌 설정
SEQUENCE_LENGTH = 20
input_features = 2 # [Length, IAT]
num_classes = 4

# 📌 1. 데이터 로드 (시퀀스 데이터)
def load_data():
    files = glob.glob("seq_features_*.csv")
    if not files:
        print("[!] No 'seq_features_*.csv' found. Please run 1extract_features.py first.")
        return None
    
    latest_file = max(files, key=os.path.getctime)
    print(f"[INFO] Loading sequence data from {latest_file}...")
    df = pd.read_csv(latest_file)
    return df

# 📌 2. LSTM 모델 정의
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # batch_first=True -> (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 초기 hidden state, cell state설정 (0으로)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 순전파
        out, _ = self.lstm(x, (h0, c0))
        
        # 마지막 타임스텝의 출력만 사용 (Many-to-One)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def main():
    df = load_data()
    if df is None: return

    # Feature/Label 분리
    # 1. Sequence Features (LSTM)
    seq_cols = [c for c in df.columns if c.startswith("pkt_")]
    X_seq_flat = df[seq_cols].values
    
    # [FIX] 입력 feature 개수 자동 계산 (20 * FEATURE_DIM)
    # 기존 코드에서 input_features=2로 고정되어 있어서 에러 발생 가능성
    # 실제 데이터 컬럼 개수로부터 역산
    actual_dim = len(seq_cols) // SEQUENCE_LENGTH
    
    # 지역 변수로 사용
    current_input_features = input_features
    if actual_dim != input_features:
        print(f"[WARN] Input features changed from {input_features} to {actual_dim}")
        current_input_features = actual_dim

    # 2. Statistical Features (Random Forest)
    stat_cols = ["p_mean", "p_std", "p_min", "p_max", "i_mean", "i_std", "i_min", "i_max", "duration", "pkt_count"]
    # 만약 예전 데이터라 컬럼이 없다면? (호환성 체크)
    if not all(col in df.columns for col in stat_cols):
        print("[ERROR] Statistical columns missing! Please re-run 1extract_features.py.")
        return

    X_stat = df[stat_cols].values
    y = df['label'].values

    # 📌 Preprocessing (LSTM)
    scaler = StandardScaler()
    X_seq_scaled_flat = scaler.fit_transform(X_seq_flat)
    
    # Reshape: (Samples, Seq_Len, Features)
    num_samples = X_seq_scaled_flat.shape[0]
    try:
        X_seq = X_seq_scaled_flat.reshape(num_samples, SEQUENCE_LENGTH, current_input_features)
    except ValueError as e:
        print(f"[FATAL] Reshape Error: {e}")
        print(f"Total elements: {X_seq_scaled_flat.size}")
        print(f"Target shape: ({num_samples}, {SEQUENCE_LENGTH}, {current_input_features}) = {num_samples*SEQUENCE_LENGTH*current_input_features}")
        return
    
    print(f"[INFO] Data Prepared. LSTM Shape: {X_seq.shape}, RF Shape: {X_stat.shape}")

    # Split (Stratified)
    # RF와 LSTM 모두 동일한 인덱스로 나눠야 비교 가능하지만, 여기서는 각각 학습해도 무방
    # 다만 편의상 X_seq를 기준으로 나누고, 같은 인덱스로 X_stat도 나눔
    indices = np.arange(num_samples)
    X_seq_train, X_seq_test, y_train, y_test, idx_train, idx_test = train_test_split(X_seq, y, indices, test_size=0.2, random_state=42, stratify=y)
    
    X_stat_train = X_stat[idx_train]
    X_stat_test = X_stat[idx_test]

    # --- [1] Random Forest Training ---
    from sklearn.ensemble import RandomForestClassifier
    print(f"[INFO] Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_stat_train, y_train)
    
    rf_pred = rf_model.predict(X_stat_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"[Result] Random Forest Accuracy: {rf_acc:.4f}")
    print(classification_report(y_test, rf_pred, labels=[0, 1, 2, 3], target_names=["Normal", "KeepAlive", "Stun", "RDP"], zero_division=0))

    # --- [2] LSTM Training ---
    # Tensor 변환
    X_train_tensor = torch.tensor(X_seq_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_seq_test, dtype=torch.float32)
    
    # 클래스 가중치 계산
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = []
    for count in class_counts:
        if count > 0:
            class_weights.append(total_samples / (len(class_counts) * count))
        else:
            class_weights.append(1.0)
    while len(class_weights) < num_classes: class_weights.append(1.0)
    
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # 모델 생성
    model = LSTMClassifier(current_input_features, 64, 2, num_classes)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습
    epochs = 50 
    history_loss = []
    history_acc = []

    print(f"[INFO] Training LSTM...")
    for epoch in range(epochs):
        model.train()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        history_loss.append(loss.item())
        
        if (epoch+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_out = model(X_test_tensor)
                _, pred = torch.max(test_out, 1)
                acc = accuracy_score(y_test, pred.numpy())
                history_acc.append(acc)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Test Acc: {acc:.4f}")

    # 최종 평가
    model.eval()
    with torch.no_grad():
        test_out = model(X_test_tensor)
        _, predicted = torch.max(test_out, 1)
        predicted = predicted.numpy()

    print("\n[LSTM Evaluation Results]")
    print(classification_report(y_test, predicted, labels=[0, 1, 2, 3], target_names=["Normal", "KeepAlive", "Stun", "RDP"], zero_division=0))

    # 저장
    torch.save(model.state_dict(), "lstm_model.pth")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(rf_model, "rf_model.pkl") # RF 모델 저장
    print("[INFO] Saved 'lstm_model.pth', 'rf_model.pkl', and 'scaler.pkl'")

    # 그래프 팝업
    try:
        plt.plot(history_loss, label='Loss')
        if history_acc:
            # acc는 10 epoch마다 찍었으므로 x축 조정 필요하지만 대충 그림
            plt.plot(np.linspace(0, epochs, len(history_acc)), history_acc, label='Accuracy')
        plt.legend()
        plt.title("LSTM Training")
        plt.show()
    except: pass

if __name__ == "__main__":
    main()
