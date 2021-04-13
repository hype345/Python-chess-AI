import chess
import chess.engine
import chess.syzygy
import chess.pgn
import chess.svg
import chess.polyglot
import asyncio
from flask import Flask, Response, request
import webbrowser
import time
import traceback
from datetime import date, datetime
import numpy as np

board = chess.Board()

# Evaluating the board
pawntable = [
    0,   0,   0,   0,   0,   0,   0,   0,
    78,  83,  86,  73, 102,  82,  85,  90,
    7,  29,  21,  44,  40,  31,  44,   7,
    -17,  16,  -2,  15,  14,   0,  15, -13,
    -26,   3,  10,   9,   6,   1,   0, -23,
    -22,   9,   5, -11, -10,  -2,   3, -19,
    -31,   8,  -7, -37, -36, -14,   3, -31,
    0,   0,   0,   0,   0,   0,   0,   0]

knightstable = [
    -66, -53, -75, -75, -10, -55, -58, -70,
    -3,  -6, 100, -36,   4,  62,  -4, -14,
    10,  67,   1,  74,  73,  27,  62,  -2,
    24,  24,  45,  37,  33,  41,  25,  17,
    -1,   5,  31,  21,  22,  35,   2,   0,
    -18,  10,  13,  22,  18,  15,  11, -14,
    -23, -15,   2,   0,   2,   0, -23, -20,
    -74, -23, -26, -24, -19, -35, -22, -69]

bishopstable = [
    -59, -78, -82, -76, -23,-107, -37, -50,
    -11,  20,  35, -42, -39,  31,   2, -22,
    -9,  39, -32,  41,  52, -10,  28, -14,
    25,  17,  20,  34,  26,  25,  15,  10,
    13,  10,  17,  23,  17,  16,   0,   7,
    14,  25,  24,  15,   8,  25,  20,  15,
    19,  20,  11,   6,   7,   6,  20,  16,
    -7,   2, -15, -12, -14, -15, -10, -10]

rookstable = [
    35,  29,  33,   4,  37,  33,  56,  50,
    55,  29,  56,  67,  55,  62,  34,  60,
    19,  35,  28,  33,  45,  27,  25,  15,
    0,   5,  16,  13,  18,  -4,  -9,  -6,
    -28, -35, -16, -21, -13, -29, -46, -30,
    -42, -28, -42, -25, -25, -35, -26, -46,
    -53, -38, -31, -26, -29, -43, -44, -53,
    -30, -24, -18,   5,  -2, -18, -31, -32]

queenstable = [
    6,   1,  -8,-104,  69,  24,  88,  26,
    14,  32,  60, -10,  20,  76,  57,  24,
    -2,  43,  32,  60,  72,  63,  43,   2,
    1, -16,  22,  17,  25,  20, -13,  -6,
    -14, -15,  -2,  -5,  -1, -10, -20, -22,
    -30,  -6, -13, -11, -16, -11, -16, -27,
    -36, -18,   0, -19, -15, -15, -21, -38,
    -39, -30, -31, -13, -31, -36, -34, -42]

kingstable = [
    4,  54,  47, -99, -99,  60,  83, -62,
    -32,  10,  55,  56,  56,  55,  10,   3,
    -62,  12, -57,  44, -67,  28,  37, -31,
    -55,  50,  11,  -4, -19,  13,   0, -49,
    -55, -43, -52, -28, -51, -47,  -8, -50,
    -47, -42, -43, -79, -64, -32, -29, -32,
    -4,   3, -14, -50, -57, -18,  13,   4,
    17,  30,  -3, -14,   6,  -1,  40,  18]


def evaluate_board():
    if board.is_checkmate():
        if board.turn:
            return -9999
        else:
            return 9999
    if board.is_stalemate():
        return 0
    if board.is_insufficient_material():
        return 0
    if num_peices() <= 5:
        try:
            with chess.syzygy.open_tablebase("/Users/cshriver/Desktop/Chess-python/Syzygy-5") as tablebase:
                    if tablebase.probe_wdl(board) > 0:
                        if board.turn:
                            return 9999
                        else:
                            return -9999
                    elif tablebase.probe_wdl(board) < 0:
                        if board.turn:
                            return -9999
                        else:
                            return 9999
                    elif tablebase.probe_wdl(board) == 0:
                        return 0
        except:
            print(num_peices(), board.fen())
            print('error position not found in table base though it should be there')

    wp = len(board.pieces(chess.PAWN, chess.WHITE))
    bp = len(board.pieces(chess.PAWN, chess.BLACK))
    wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
    wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    bb = len(board.pieces(chess.BISHOP, chess.BLACK))
    wr = len(board.pieces(chess.ROOK, chess.WHITE))
    br = len(board.pieces(chess.ROOK, chess.BLACK))
    wq = len(board.pieces(chess.QUEEN, chess.WHITE))
    bq = len(board.pieces(chess.QUEEN, chess.BLACK))

    material = 100 * (wp - bp) + 280 * (wn - bn) + 320 * (wb - bb) + 479 * (wr - br) + 929 * (wq - bq)

    pawnsq = sum([pawntable[i] for i in board.pieces(chess.PAWN, chess.WHITE)])
    pawnsq = pawnsq + sum([-pawntable[chess.square_mirror(i)]
                        for i in board.pieces(chess.PAWN, chess.BLACK)])
    knightsq = sum([knightstable[i] for i in board.pieces(chess.KNIGHT, chess.WHITE)])
    knightsq = knightsq + sum([-knightstable[chess.square_mirror(i)]
                            for i in board.pieces(chess.KNIGHT, chess.BLACK)])
    bishopsq = sum([bishopstable[i] for i in board.pieces(chess.BISHOP, chess.WHITE)])
    bishopsq = bishopsq + sum([-bishopstable[chess.square_mirror(i)]
                            for i in board.pieces(chess.BISHOP, chess.BLACK)])
    rooksq = sum([rookstable[i] for i in board.pieces(chess.ROOK, chess.WHITE)])
    rooksq = rooksq + sum([-rookstable[chess.square_mirror(i)]
                        for i in board.pieces(chess.ROOK, chess.BLACK)])
    queensq = sum([queenstable[i] for i in board.pieces(chess.QUEEN, chess.WHITE)])
    queensq = queensq + sum([-queenstable[chess.square_mirror(i)]
                            for i in board.pieces(chess.QUEEN, chess.BLACK)])
    kingsq = sum([kingstable[i] for i in board.pieces(chess.KING, chess.WHITE)])
    kingsq = kingsq + sum([-kingstable[chess.square_mirror(i)]
                        for i in board.pieces(chess.KING, chess.BLACK)])

    eval = material + pawnsq + knightsq + bishopsq + rooksq + queensq + kingsq
    return eval

# find number of piece left on board to see if we should look at tablebases
def num_peices():
    total = 2
    for i in board.pieces(chess.QUEEN, chess.WHITE):
        if i is not None:
            total +=1
    for i in board.pieces(chess.QUEEN, chess.BLACK):
        if i is not None:
            total +=1
    for i in board.pieces(chess.ROOK, chess.WHITE):
        if i is not None:
            total +=1
    for i in board.pieces(chess.ROOK, chess.BLACK):
        if i is not None:
            total +=1
    for i in board.pieces(chess.KNIGHT, chess.WHITE):
        if i is not None:
            total +=1
    for i in board.pieces(chess.KNIGHT, chess.BLACK):
        if i is not None:
            total +=1
    for i in board.pieces(chess.BISHOP, chess.WHITE):
        if i is not None:
            total +=1
    for i in board.pieces(chess.BISHOP, chess.BLACK):
        if i is not None:
            total +=1
    for i in board.pieces(chess.PAWN, chess.WHITE):
        if i is not None:
            total +=1
    for i in board.pieces(chess.PAWN, chess.BLACK):
        if i is not None:
            total +=1

    return total

# Orders moves to speed up search through better prunning
def moveOrdering(moves):
    result = {}
    pieceVaules = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}
    for move in moves:
        moveSore = 0
        if board.is_capture(move):
            if board.is_en_passant(move):
                moveSore += 20
            else:
                moveSore += (10 * (pieceVaules[board.piece_at(move.to_square).piece_type] - pieceVaules[board.piece_at(move.from_square).piece_type])) + 5
        if move.promotion is not None:
            moveSore += pieceVaules[move.promotion] * 7
        result[move] = moveSore
    result = sorted(result.items(), key=lambda x: x[1], reverse=True)
    finalResult = []
    for obj in result:
        finalResult.append(obj[0])
    return finalResult


# Searching the best move using minimax and alphabeta algorithm with negamax implementation
def alphabeta(alpha, beta, depthleft):
    bestscore = -9999
    if (depthleft == 0):
        return quiesce(alpha, beta)
    moves = moveOrdering(board.legal_moves)
    for move in moves:
        board.push(move)
        score = -alphabeta(-beta, -alpha, depthleft - 1)
        board.pop()
        if (score >= beta):
            return score
        if (score > bestscore):
            bestscore = score
        if (score > alpha):
            alpha = score
    return bestscore

# Before evauling board finds quite possition (No captures)
def quiesce(alpha, beta):
    if board.turn:
        stand_pat = evaluate_board()
    else:
        stand_pat = -(evaluate_board())
        
    if (stand_pat >= beta):
        return beta
    if (alpha < stand_pat):
        alpha = stand_pat
    moves = moveOrdering(board.legal_moves)
    for move in moves:
        if board.is_capture(move):
            board.push(move)
            score = -quiesce(-beta, -alpha)
            board.pop()

            if (score >= beta):
                return beta
            if (score > alpha):
                alpha = score
    return alpha


def selectmove(depth):
    try:
        move = chess.polyglot.MemoryMappedReader("/Users/cshriver/Desktop/Chess-python/books/human.bin").weighted_choice(board).move
        return move
    except:
        if num_peices() <= 5:
            try:
                with chess.syzygy.open_tablebase("/Users/cshriver/Desktop/Chess-python/Syzygy-5") as tablebase:
                    cur_dtz = tablebase.probe_dtz(board)
                    fastest_win = -9999
                    bestMove = chess.Move.null()
                    for move in board.legal_moves:
                        board.push(move)
                        if cur_dtz > 0:
                            if tablebase.probe_dtz(board) < cur_dtz and tablebase.probe_dtz(board) < 0:
                                if fastest_win < tablebase.probe_dtz(board):
                                    fastest_win = tablebase.probe_dtz(board)
                                    bestMove = move
                        elif cur_dtz < 0:
                            if tablebase.probe_dtz(board) == (abs(cur_dtz) - 1):
                                bestMove = move
                        elif cur_dtz == 0:
                            if tablebase.probe_dtz(board) == 0:
                                bestMove = move
                        board.pop()
                return bestMove
            except:
                print('error position not found in table base')

        bestMove = chess.Move.null()
        bestValue = -99999
        alpha = -100000
        beta = 100000
        moves = moveOrdering(board.legal_moves)
        for move in moves:
            board.push(move)
            boardValue = -alphabeta(-beta, -alpha, depth - 1)
            if boardValue > bestValue:
                bestValue = boardValue
                bestMove = move
            if (boardValue > alpha):
                alpha = boardValue
            board.pop()
        return bestMove


# Searching alphafish's Move
def alphafish_move():
    if not board.is_game_over(claim_draw=True):
        move = selectmove(4)
        moveHistory.append(str(move))
        board.push(move)
    else:
        print(board.result())

# Searching Stockfish's Move
async def stockfish() -> None:
    if not board.is_game_over(claim_draw=True):
        transport, engine = await chess.engine.popen_uci("/usr/local/Cellar/stockfish/13/bin/stockfish")
        result = await engine.play(board, chess.engine.Limit(time=0.1))
        moveHistory.append(str(result.move))
        board.push(result.move)
        await engine.quit()
    else:
        print(board.result())

def pgn_generator(myBoard=chess.Board()):
    global PGN
    now = datetime.now()
    game = chess.pgn.Game()
    game.setup(myBoard)
    first = True

    game.headers['Event'] = 'online'
    game.headers['Site'] = 'python-chess'
    game.headers['Date'] = now.strftime("%d/%m/%Y %H:%M:%S")
    game.headers['Round'] = 'Single Game'
    game.headers['White'] = '?'
    game.headers['Black'] = '?'
    game.headers['Result'] = board.result() 


    for move in moveHistory:
        try:
            if first:
                node = game.add_variation(chess.Move.from_uci(move))
                first =False
            else:
                node = node.add_variation(chess.Move.from_uci(move))
        except:
            print('error with: ',move)

    PGN = game

def reset_moveHistory():
    global moveHistory
    moveHistory = []

def moveHistory_pop():
    global moveHistory
    moveHistory.pop()

def moveHistory_push(move):
    global moveHistory
    moveHistory.append(board.uci(board.parse_san(move)))

def set_result():
    global Result
    Result = board.result()

def store_FEN(fen):
    global stored_FEN
    stored_FEN =  fen


app = Flask(__name__)


# Front Page of the Flask Web Page
@app.route("/")
def main():
    global board
    ret = '<html><head>'
    ret += '<style>input {font-size: 20px; } button { font-size: 20px; }</style>'
    ret += '</head><body>'
    ret += str(Result)
    ret += '</br>'
    ret += '<img width=410 height=410 src="/board.svg?%f"></img></br>' % time.time()
    ret +=  str(moveHistory)
    ret += '</br>'
    ret += '<form action="/game/" method="post"><button name="New Game" type="submit">New Game</button></form>'
    ret += '<form action="/load/"><input type="submit" value="Load FEN:"><input name="FEN" type="text"></input></form>'
    ret += '<form action="/undo/" method="post"><button name="Undo" type="submit">Undo Last Move</button></form>'
    ret += '<form action="/move/"><input type="submit" value="Make Human Move:"><input name="move" type="text"></input></form>'
    ret += '<form action="/recv/" method="post"><button name="Receive Move" type="submit">Receive Human Move</button></form>'
    ret += '<form action="/alphafish/" method="post"><button name="Comp Move" type="submit">Make Alphafish Move</button></form>'
    ret += '<form action="/stockfish/" method="post"><button name="Stockfish Move" type="submit">Make Stockfish Move</button></form>'
    ret += '<form action="/get_pgn/" method="post"><button name="Get PGN" type="submit">Get PGN</button></form>'
    ret += str(PGN) 
    if board.is_stalemate():
        print("Its a draw by stalemate")
    elif board.is_checkmate():
        print("Checkmate")
    elif board.is_insufficient_material():
        print("Its a draw by insufficient material")
    return ret

# Display Board
@app.route("/board.svg/")
def board():
    return Response(chess.svg.board(board=board, size=700), mimetype='image/svg+xml')

# Human Move
@app.route("/move/")
def move():
    try:
        move = request.args.get('move', default="")
        if not board.is_game_over(claim_draw=True):
            moveHistory_push(move)
            board.push_san(move)
        set_result()
    except Exception:
        traceback.print_exc()
    return main()

# Recieve Human Move
@app.route("/recv/", methods=['POST'])
def recv():
    try:
        None
    except Exception:
        None
    return main()

# Make Alphafish-Zero Move
@app.route("/alphafish/", methods=['POST'])
def alphafish():
    try:
        alphafish_move()
        set_result()
    except Exception:
        traceback.print_exc()
    return main()

# Make UCI Compatible engine's move
@app.route("/stockfish/", methods=['POST'])
def engine():
    try:
        asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
        asyncio.run(stockfish())
        set_result()
    except Exception:
        traceback.print_exc()
    return main()

# New Game
@app.route("/game/", methods=['POST'])
def game():
    board.reset()
    reset_moveHistory()
    set_result()
    return main()

# Display's PGN
@app.route("/get_pgn/", methods=['POST'])
def get_pgn():
    pgn_generator(stored_FEN)
    set_result()
    return main()

# Load FEN Game
@app.route("/load/")
def load():
    try:
        FEN = request.args.get('FEN', default="")
        reset_moveHistory()
        board.set_fen(FEN)
        store_FEN(FEN)
        set_result()
    except Exception:
        traceback.print_exc()
    return main()


# Undo
@app.route("/undo/", methods=['POST'])
def undo():
    try:
        board.pop()
        moveHistory_pop()
        set_result()
    except Exception:
        traceback.print_exc()
    return main()



# Main Function
if __name__ == '__main__':
    board = chess.Board()
    moveHistory = []
    stored_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
    PGN = None
    Result = None
    # positionsSearched = 0
    webbrowser.open("http://127.0.0.1:5000/")
    app.run()

# python main.py