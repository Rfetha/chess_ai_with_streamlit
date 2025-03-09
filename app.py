# app.py
import sys
import torch

# Torch özel hata düzeltmesi
sys.modules['torch.classes'] = None
sys.modules['torch._classes'] = None

import streamlit as st
import chess
import chess.svg
import numpy as np
import base64
import pickle
from chess import Board
from model import ChessModel
from functions import board_as_matrix

# Cihaz konfigürasyonu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    """Model ve sözlükleri yükler"""
    with open("models/heavy_move_to_int.pkl", "rb") as file:
        move_to_int = pickle.load(file)
    int_to_move = {v: k for k, v in move_to_int.items()}
    
    model = ChessModel(num_classes=len(move_to_int))
    model.load_state_dict(torch.load("models/TORCH_50EPOCHS.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, int_to_move

def prepare_input(board: Board):
    """Tahtayı tensor formatına çevirir"""
    matrix = board_as_matrix(board)
    return torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)

def predict_move(board: Board, model, int_to_move):
    """Yapay zeka hamlesini üretir"""
    X_tensor = prepare_input(board).to(device)
    
    with torch.no_grad():
        logits = model(X_tensor)
    
    probs = torch.softmax(logits.squeeze(0), dim=0).cpu().numpy()
    legal_moves = [move.uci() for move in board.legal_moves]
    
    for move_idx in np.argsort(probs)[::-1]:
        move = int_to_move[move_idx]
        if move in legal_moves:
            return move
    return None

def display_board(board: Board):
    """Tahtayı interaktif SVG olarak göster"""
    svg = chess.svg.board(
        board=board,
        size=600,
        lastmove=board.peek() if len(board.move_stack) > 0 else None
    )
    b64 = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
    html = f'<img src="data:image/svg+xml;base64,{b64}"/>'
    st.write(html, unsafe_allow_html=True, key=board.fen())

def handle_move(move_uci: str, model, int_to_move):
    """Hamleleri yönetir"""
    try:
        move = st.session_state.board.parse_uci(move_uci.strip())
        if move not in st.session_state.board.legal_moves:
            st.error("❌ Geçersiz hamle!")
            return  # Tahtayı kaldırma, sadece hata mesajını göster
        
        # Kullanıcı hamlesi
        st.session_state.board.push(move)
        
        # AI hamlesi
        if not st.session_state.board.is_game_over():
            ai_move = predict_move(st.session_state.board, model, int_to_move)
            if ai_move:
                st.session_state.board.push(chess.Move.from_uci(ai_move))
        
        st.rerun()  # Hata olmadığında tahtayı güncelle
    except ValueError:
        st.error("❌ Hatalı format! Örnek: e2e4")
    except IndexError:
        st.error("⚠️ Lütfen hamle girin!")

def main():
    # Sayfa konfigürasyonu
    st.set_page_config(page_title="Satranç AI", layout="wide")
    st.title("♟️ Satranç Yapay Zeka Uygulaması")
    
    # Model yükleme
    model, int_to_move = load_model()
    
    # Oyun durumu
    if 'board' not in st.session_state:
        st.session_state.board = Board()
        st.session_state.last_fen = ""
    
    # Ana arayüz
    # Container ile ortalama işlemi
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # 🛠️ Hata oluşsa bile tahtayı her zaman göster
            display_board(st.session_state.board)
        
        with col2:
            # Kontrol paneli
            with st.form("move_form"):
                user_move = st.text_input("UCI Hamle:", key="move_input", 
                                          placeholder="e2e4, g1f3 vb...")
                
                submitted = st.form_submit_button("Hamle Yap 🚀")
                if submitted:
                    handle_move(user_move, model, int_to_move)
            
            # Oyun kontrolleri
            st.markdown("---")
            if st.button("🔄 Yeni Oyun"):
                st.session_state.board.reset()
                st.session_state.last_fen = ""
                st.rerun()
            
            if st.session_state.board.is_game_over():
                st.success(f"🏆 Oyun Sonu: {st.session_state.board.result()}")
                st.balloons()

if __name__ == "__main__":
    main()