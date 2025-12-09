import { useEffect, useState, useCallback } from 'react';
import type { Square, Side, Formation } from './types';
import { DEFAULT_FORMATION } from './constants';
import { useToast } from './hooks/useToast';
import { useGameApi } from './hooks/useGameApi';
import { useCanvas } from './hooks/useCanvas';
import { squareToNotation } from './utils';
import { BoardCanvas } from './components/BoardCanvas';
import { GameControls } from './components/GameControls';
import { GameStatus } from './components/GameStatus';
import { SetupPanel } from './components/SetupPanel';
import { MoveHistory } from './components/MoveHistory';
import { ToastContainer } from './components/Toast';
import '../GamePage.css';

const GamePage = () => {
  const { toasts, showToast, removeToast } = useToast();
  const {
    boardData,
    boardKorean,
    legalMoves,
    isGameOver,
    canUndo,
    inOpeningBook,
    moveHistory,
    status,
    statusClass,
    inCheckSquare,
    lastMoveFrom,
    lastMoveTo,
    initGame,
    makeMove,
    aiMove,
    undoMovePair,
  } = useGameApi(showToast);

  const [selectedSquare, setSelectedSquare] = useState<Square | null>(null);
  const [possibleMoveSquares, setPossibleMoveSquares] = useState<Square[]>([]);
  const [hoverSquare, setHoverSquare] = useState<Square | null>(null);
  const [boardFlipped, setBoardFlipped] = useState(false);
  const [mySide, setMySide] = useState<Side>('han');
  const [isAutoPlaying, setIsAutoPlaying] = useState(false);
  const [showSetupPanel, setShowSetupPanel] = useState(false);
  const [myFormation, setMyFormation] = useState<Formation>(DEFAULT_FORMATION);
  const [opponentFormation, setOpponentFormation] = useState<Formation>(DEFAULT_FORMATION);
  const [showMoveInfo, setShowMoveInfo] = useState(false);
  const [selectedPiece, setSelectedPiece] = useState('');
  const [possibleMoves, setPossibleMoves] = useState('');

  const {
    canvasRef,
    canvasWidth,
    canvasHeight,
    fileRankToPixel,
    getCanvasCoordinates,
  } = useCanvas(boardFlipped);

  // 내 진영이 아래쪽에 오도록 보드 자동 뒤집기
  useEffect(() => {
    setBoardFlipped(mySide === 'cho');
  }, [mySide]);

  // 게임 초기화
  useEffect(() => {
    initGame();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const getPieceAt = useCallback(
    (file: number, rank: number): string | null => {
      const rankIdx = 9 - rank;
      return boardData[rankIdx]?.[file] || null;
    },
    [boardData]
  );

  const selectSquare = useCallback(
    (file: number, rank: number) => {
      const piece = getPieceAt(file, rank);
      if (!piece) return;

      setSelectedSquare({ file, rank });
      const squareKey = squareToNotation(file, rank);
      const movesFromSquare = legalMoves.filter((m: { from: string; to: string }) => m.from === squareKey);
      setPossibleMoveSquares(
        movesFromSquare.map((m: { from: string; to: string }) => {
          const toFile = m.to.charCodeAt(0) - 97;
          const toRank = parseInt(m.to.slice(1)) - 1;
          return { file: toFile, rank: toRank };
        })
      );

      const rankIdx = 9 - rank;
      const korean = boardKorean[rankIdx]?.[file];
      if (korean) {
        setShowMoveInfo(true);
        setSelectedPiece(korean.full_name || korean.name);
        setPossibleMoves(
          movesFromSquare.length > 0
            ? movesFromSquare.map((m: { from: string; to: string }) => m.to).join(', ')
            : '없음'
        );
      }
    },
    [legalMoves, boardKorean, getPieceAt]
  );

  const clearSelection = useCallback(() => {
    setSelectedSquare(null);
    setPossibleMoveSquares([]);
    setShowMoveInfo(false);
  }, []);

  const handleMakeMove = useCallback(
    async (fromFile: number, fromRank: number, toFile: number, toRank: number) => {
      const success = await makeMove(fromFile, fromRank, toFile, toRank);
      if (success) {
        clearSelection();
      }
    },
    [makeMove, clearSelection]
  );

  const handleAiMove = useCallback(async () => {
    const result = await aiMove();
    if (result && result.gameOver) {
      setIsAutoPlaying(false);
    }
  }, [aiMove]);

  const toggleAutoPlay = useCallback(async () => {
    if (isGameOver) {
      showToast('게임이 종료되었습니다. 새 게임을 시작하세요.', 'warning');
      return;
    }

    if (isAutoPlaying) {
      setIsAutoPlaying(false);
      showToast('AI 자동 진행이 중지되었습니다.', 'info');
    } else {
      setIsAutoPlaying(true);
      showToast('AI 자동 진행을 시작합니다.', 'info');
    }
  }, [isGameOver, isAutoPlaying, showToast]);

  // 자동 진행 루프
  useEffect(() => {
    if (!isAutoPlaying || isGameOver) return;

    let cancelled = false;

    const autoPlayLoop = async () => {
      while (!cancelled && isAutoPlaying && !isGameOver) {
        const result = await aiMove();
        if ((result && result.gameOver) || cancelled) {
          setIsAutoPlaying(false);
          break;
        }
        if (!cancelled) {
          await new Promise((resolve) => setTimeout(resolve, 500));
        }
      }
    };

    autoPlayLoop();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAutoPlaying, isGameOver]);

  const handleUndo = useCallback(async () => {
    if (isAutoPlaying) {
      showToast('자동 진행 중에는 되돌릴 수 없습니다.', 'warning');
      return;
    }
    await undoMovePair();
    clearSelection();
  }, [undoMovePair, isAutoPlaying, clearSelection, showToast]);

  const applyFormation = useCallback(async () => {
    const flipFormation = (formation: string) => {
      const flipMap: Record<string, string> = {
        '상마상마': '마상마상',
        '마상마상': '상마상마',
        '마상상마': '마상상마',
        '상마마상': '상마마상',
      };
      return flipMap[formation] || formation;
    };

    let hanFormation: string, choFormation: string;

    if (mySide === 'cho') {
      choFormation = myFormation;
      hanFormation = flipFormation(opponentFormation);
    } else {
      hanFormation = myFormation;
      choFormation = flipFormation(opponentFormation);
    }

    await initGame(null, hanFormation, choFormation);
    setShowSetupPanel(false);
    const mySideLabel = mySide === 'cho' ? '초' : '한';
    const opponentSideLabel = mySide === 'cho' ? '한' : '초';
    showToast(
      `${mySideLabel}: ${myFormation}, ${opponentSideLabel}: ${opponentFormation} 상차림으로 게임을 시작합니다!`,
      'success'
    );
  }, [mySide, myFormation, opponentFormation, initGame, showToast]);

  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (isGameOver) return;

      const pos = getCanvasCoordinates(e);
      if (!pos) return;

      const { file, rank } = pos;

      if (selectedSquare) {
        const isPossibleMove = possibleMoveSquares.some(
          (sq) => sq.file === file && sq.rank === rank
        );

        if (isPossibleMove) {
          handleMakeMove(selectedSquare.file, selectedSquare.rank, file, rank);
        } else if (file === selectedSquare.file && rank === selectedSquare.rank) {
          clearSelection();
        } else {
          const piece = getPieceAt(file, rank);
          if (piece) {
            selectSquare(file, rank);
          } else {
            clearSelection();
          }
        }
      } else {
        selectSquare(file, rank);
      }
    },
    [isGameOver, selectedSquare, possibleMoveSquares, getCanvasCoordinates, handleMakeMove, clearSelection, getPieceAt, selectSquare]
  );

  const handleCanvasMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const pos = getCanvasCoordinates(e);
      if (pos) {
        setHoverSquare({ file: pos.file, rank: pos.rank });
      } else {
        setHoverSquare(null);
      }
    },
    [getCanvasCoordinates]
  );

  const handleCanvasMouseLeave = useCallback(() => {
    setHoverSquare(null);
  }, []);

  return (
    <div className="container">
      <h1>장기 AI 엔진</h1>

      <GameControls
        onNewGame={() => initGame()}
        onAiMove={handleAiMove}
        onToggleAutoPlay={toggleAutoPlay}
        onUndo={handleUndo}
        onToggleSetup={() => setShowSetupPanel(!showSetupPanel)}
        onToggleFlip={() => setBoardFlipped(!boardFlipped)}
        isGameOver={isGameOver}
        isAutoPlaying={isAutoPlaying}
        canUndo={canUndo}
        boardFlipped={boardFlipped}
      />

      <GameStatus
        status={status}
        statusClass={statusClass}
        inOpeningBook={inOpeningBook}
        hasBoard={boardData.length > 0}
      />

      <div className="main-content">
        <div className="board-container">
          <div className="board-wrapper">
            <canvas
              ref={canvasRef}
              width={canvasWidth}
              height={canvasHeight}
              onClick={handleCanvasClick}
              onMouseMove={handleCanvasMouseMove}
              onMouseLeave={handleCanvasMouseLeave}
            />
            <BoardCanvas
              canvasRef={canvasRef}
              canvasWidth={canvasWidth}
              canvasHeight={canvasHeight}
              boardData={boardData}
              boardKorean={boardKorean}
              selectedSquare={selectedSquare}
              possibleMoveSquares={possibleMoveSquares}
              lastMoveFrom={lastMoveFrom}
              lastMoveTo={lastMoveTo}
              inCheckSquare={inCheckSquare}
              hoverSquare={hoverSquare}
              boardFlipped={boardFlipped}
              fileRankToPixel={fileRankToPixel}
              getPieceAt={getPieceAt}
            />
          </div>
          {showMoveInfo && (
            <div className="move-info">
              <strong>선택한 기물:</strong> <span>{selectedPiece}</span>
              <br />
              <strong>가능한 이동:</strong> <span>{possibleMoves}</span>
            </div>
          )}
        </div>

        {showSetupPanel && (
          <SetupPanel
            mySide={mySide}
            myFormation={myFormation}
            opponentFormation={opponentFormation}
            onMySideChange={setMySide}
            onMyFormationChange={setMyFormation}
            onOpponentFormationChange={setOpponentFormation}
            onApply={applyFormation}
          />
        )}

        <MoveHistory moveHistory={moveHistory} />
      </div>

      <ToastContainer toasts={toasts} onClose={removeToast} />
    </div>
  );
};

export default GamePage;

