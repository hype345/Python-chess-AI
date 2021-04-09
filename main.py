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
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0]

knightstable = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50]

bishopstable = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20]

rookstable = [
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0]

queenstable = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 5, 5, 5, 5, 5, 0, -10,
    0, 0, 5, 5, 5, 5, 0, -5,
    -5, 0, 5, 5, 5, 5, 0, -5,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20]

kingstable = [
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30]


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

    material = 100 * (wp - bp) + 320 * (wn - bn) + 330 * (wb - bb) + 500 * (wr - br) + 900 * (wq - bq)

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
    if board.turn:
        return eval
    else:
        return -eval

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


# Searching the best move using minimax and alphabeta algorithm with negamax implementation
def alphabeta(alpha, beta, depthleft):
    # global positionsSearched
    bestscore = -9999
    if (depthleft == 0):
        return quiesce(alpha, beta)
    for move in board.legal_moves:
        board.push(move)
        # positionsSearched += 1
        score = -alphabeta(-beta, -alpha, depthleft - 1)
        board.pop()
        if (score >= beta):
            return score
        if (score > bestscore):
            bestscore = score
        if (score > alpha):
            alpha = score
    return bestscore


def quiesce(alpha, beta):
    # global positionsSearched
    stand_pat = evaluate_board()
    if (stand_pat >= beta):
        return beta
    if (alpha < stand_pat):
        alpha = stand_pat

    for move in board.legal_moves:
        if board.is_capture(move):
            board.push(move)
            # positionsSearched += 1
            score = -quiesce(-beta, -alpha)
            board.pop()

            if (score >= beta):
                return beta
            if (score > alpha):
                alpha = score
    return alpha


def selectmove(depth):
    # global positionsSearched
    try:
        move = chess.polyglot.MemoryMappedReader("/Users/cshriver/Desktop/Chess-python/books/human.bin").weighted_choice(board).move
        # move = chess.polyglot.MemoryMappedReader("/Users/cshriver/Desktop/Chess-python/books/computer.bin").weighted_choice(board).move
        # move = chess.polyglot.MemoryMappedReader("/Users/cshriver/Desktop/Chess-python/books/human.bin").weighted_choice(board).move
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
                        # positionsSearched += 1
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
        for move in board.legal_moves:
            board.push(move)
            # positionsSearched += 1
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
        move = selectmove(3)
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
# app.config['DEBUG'] = True


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
    # ret += str(positionsSearched)
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
        # global positionsSearched
        # positionsSearched = 0
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