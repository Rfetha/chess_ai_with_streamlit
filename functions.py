import numpy as np
from chess import Board


import numpy as np
import chess

def board_as_matrix(board: chess.Board):
    matrix = np.zeros((13, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()

    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        color_offset = 0 if piece.color == chess.WHITE else 6
        matrix[piece_type + color_offset, row, col] = 1

    for move in board.legal_moves:
        to_square = move.to_square
        row, col = divmod(to_square, 8) 
        matrix[12, row, col] = 1

    return matrix

def create_input_for_nn(games):
    X = []
    y = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            X.append(board_as_matrix(board))
            y.append(move.uci())
            board.push(move)
    return np.array(X, dtype=np.float32), np.array(y)


def encode_moves(moves):
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return np.array([move_to_int[move] for move in moves], dtype=np.float32), move_to_int
