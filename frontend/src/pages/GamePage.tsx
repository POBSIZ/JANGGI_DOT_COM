import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import './GamePage.css';

const API_BASE = '/api';
const GAME_ID = 'default';

interface BoardData {
  board: string[][];
  board_korean: Array<Array<{ name: string; full_name?: string }>>;
  legal_moves: Array<{ from: string; to: string }>;
  side_to_move: 'HAN' | 'CHO';
  game_over: boolean;
  winner: 'HAN' | 'CHO' | null;
  in_check: boolean;
  can_undo: boolean;
  in_opening_book: boolean;
  move_history: Array<{ move_number: number; notation: string; captured?: boolean }>;
  draw_reason?: string;
}

interface Square {
  file: number;
  rank: number;
}

const GamePage = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [boardData, setBoardData] = useState<string[][]>([]);
  const [boardKorean, setBoardKorean] = useState<Array<Array<{ name: string; full_name?: string }>>>([]);
  const [legalMoves, setLegalMoves] = useState<Array<{ from: string; to: string }>>([]);
  const [selectedSquare, setSelectedSquare] = useState<Square | null>(null);
  const [possibleMoveSquares, setPossibleMoveSquares] = useState<Square[]>([]);
  const [lastMoveFrom, setLastMoveFrom] = useState<Square | null>(null);
  const [lastMoveTo, setLastMoveTo] = useState<Square | null>(null);
  const [inCheckSquare, setInCheckSquare] = useState<Square | null>(null);
  const [hoverSquare, setHoverSquare] = useState<Square | null>(null);
  const [boardFlipped, setBoardFlipped] = useState(false);
  const [mySide, setMySide] = useState<'cho' | 'han'>('han');
  const [isGameOver, setIsGameOver] = useState(false);
  const [isAutoPlaying, setIsAutoPlaying] = useState(false);
  const [canUndo, setCanUndo] = useState(false);
  const [inOpeningBook, setInOpeningBook] = useState(false);
  const [status, setStatus] = useState('게임을 시작하세요');
  const [statusClass, setStatusClass] = useState('status info');
  const [moveHistory, setMoveHistory] = useState<Array<{ move_number: number; notation: string; captured?: boolean }>>([]);
  const [showSetupPanel, setShowSetupPanel] = useState(false);
  const [myFormation, setMyFormation] = useState('마상상마');
  const [opponentFormation, setOpponentFormation] = useState('마상상마');
  const [toasts, setToasts] = useState<Array<{ id: number; message: string; type: string }>>([]);
  const [showMoveInfo, setShowMoveInfo] = useState(false);
  const [selectedPiece, setSelectedPiece] = useState('');
  const [possibleMoves, setPossibleMoves] = useState('');

  // Canvas dimensions
  const BASE_CANVAS_WIDTH = 520;
  const BASE_CANVAS_HEIGHT = 580;
  const [canvasWidth, setCanvasWidth] = useState(BASE_CANVAS_WIDTH);
  const [canvasHeight, setCanvasHeight] = useState(BASE_CANVAS_HEIGHT);
  const PADDING = 40;
  const CELL_WIDTH = (canvasWidth - PADDING * 2) / 8;
  const CELL_HEIGHT = (canvasHeight - PADDING * 2) / 9;
  const PIECE_RADIUS = 24;

  const COLORS = {
    boardBg: '#c9a66b',
    boardLine: '#3d2914',
    palaceLine: '#3d2914',
    pieceBase: '#faf6e8',
    pieceBorder: '#2d2d2d',
    hanColor: '#c41e3a',
    choColor: '#1a5fb4',
    selectedGlow: '#90ee90',
    possibleMove: 'rgba(74, 144, 226, 0.6)',
    captureMove: 'rgba(220, 53, 69, 0.7)',
    lastMoveFrom: 'rgba(255, 215, 0, 0.4)',
    lastMoveTo: 'rgba(50, 205, 50, 0.4)',
    checkGlow: '#ff0000',
    hoverGlow: 'rgba(255, 255, 255, 0.4)',
  };

  useEffect(() => {
    initGame();
    resizeCanvas();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    if (canvasRef.current && boardData.length > 0) {
      drawBoard();
    }
  }, [boardData, selectedSquare, possibleMoveSquares, lastMoveFrom, lastMoveTo, inCheckSquare, hoverSquare, boardFlipped]);

  const handleResize = () => {
    resizeCanvas();
  };

  const resizeCanvas = () => {
    const container = document.querySelector('.board-wrapper');
    if (!container) return;
    
    const maxWidth = Math.min((container as HTMLElement).clientWidth - 40, BASE_CANVAS_WIDTH);
    const scaleFactor = maxWidth / BASE_CANVAS_WIDTH;
    setCanvasWidth(maxWidth);
    setCanvasHeight(BASE_CANVAS_HEIGHT * scaleFactor);
    
    if (canvasRef.current) {
      canvasRef.current.width = maxWidth;
      canvasRef.current.height = BASE_CANVAS_HEIGHT * scaleFactor;
      canvasRef.current.style.width = maxWidth + 'px';
      canvasRef.current.style.height = BASE_CANVAS_HEIGHT * scaleFactor + 'px';
    }
  };

  const fileRankToPixel = (file: number, rank: number) => {
    const displayFile = boardFlipped ? (8 - file) : file;
    const displayRank = boardFlipped ? (9 - rank) : rank;
    const x = PADDING + displayFile * CELL_WIDTH;
    const y = PADDING + (9 - displayRank) * CELL_HEIGHT;
    return { x, y };
  };

  const pixelToFileRank = (px: number, py: number): Square | null => {
    const displayFile = Math.round((px - PADDING) / CELL_WIDTH);
    const rankFromTop = Math.round((py - PADDING) / CELL_HEIGHT);
    const displayRank = 9 - rankFromTop;
    const file = boardFlipped ? (8 - displayFile) : displayFile;
    const rank = boardFlipped ? (9 - displayRank) : displayRank;

    if (file >= 0 && file <= 8 && rank >= 0 && rank <= 9) {
      return { file, rank };
    }
    return null;
  };

  const getPieceAt = (file: number, rank: number): string | null => {
    const rankIdx = 9 - rank;
    return boardData[rankIdx]?.[file] || null;
  };

  const drawBoard = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = COLORS.boardBg;
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    ctx.strokeStyle = COLORS.boardLine;
    ctx.lineWidth = 1.5;

    // Draw horizontal lines
    for (let i = 0; i < 10; i++) {
      const y = PADDING + i * CELL_HEIGHT;
      ctx.beginPath();
      ctx.moveTo(PADDING, y);
      ctx.lineTo(canvasWidth - PADDING, y);
      ctx.stroke();
    }

    // Draw vertical lines
    for (let i = 0; i < 9; i++) {
      const x = PADDING + i * CELL_WIDTH;
      ctx.beginPath();
      ctx.moveTo(x, PADDING);
      ctx.lineTo(x, canvasHeight - PADDING);
      ctx.stroke();
    }

    // Draw palace diagonals
    ctx.strokeStyle = COLORS.palaceLine;
    const topPalace = {
      left: fileRankToPixel(3, 9),
      right: fileRankToPixel(5, 9),
      bottomLeft: fileRankToPixel(3, 7),
      bottomRight: fileRankToPixel(5, 7),
    };
    ctx.beginPath();
    ctx.moveTo(topPalace.left.x, topPalace.left.y);
    ctx.lineTo(topPalace.bottomRight.x, topPalace.bottomRight.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(topPalace.right.x, topPalace.right.y);
    ctx.lineTo(topPalace.bottomLeft.x, topPalace.bottomLeft.y);
    ctx.stroke();

    const bottomPalace = {
      left: fileRankToPixel(3, 2),
      right: fileRankToPixel(5, 2),
      bottomLeft: fileRankToPixel(3, 0),
      bottomRight: fileRankToPixel(5, 0),
    };
    ctx.beginPath();
    ctx.moveTo(bottomPalace.left.x, bottomPalace.left.y);
    ctx.lineTo(bottomPalace.bottomRight.x, bottomPalace.bottomRight.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(bottomPalace.right.x, bottomPalace.right.y);
    ctx.lineTo(bottomPalace.bottomLeft.x, bottomPalace.bottomLeft.y);
    ctx.stroke();

    // Draw highlights
    if (lastMoveFrom) {
      const { x, y } = fileRankToPixel(lastMoveFrom.file, lastMoveFrom.rank);
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, PIECE_RADIUS + 10);
      gradient.addColorStop(0, 'rgba(255, 215, 0, 0.3)');
      gradient.addColorStop(1, 'rgba(255, 215, 0, 0)');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, PIECE_RADIUS + 10, 0, Math.PI * 2);
      ctx.fill();
    }

    if (lastMoveTo) {
      const { x, y } = fileRankToPixel(lastMoveTo.file, lastMoveTo.rank);
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, PIECE_RADIUS + 10);
      gradient.addColorStop(0, 'rgba(50, 205, 50, 0.3)');
      gradient.addColorStop(1, 'rgba(50, 205, 50, 0)');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, PIECE_RADIUS + 10, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw possible moves
    possibleMoveSquares.forEach((sq) => {
      const { x, y } = fileRankToPixel(sq.file, sq.rank);
      const hasPiece = getPieceAt(sq.file, sq.rank);
      if (hasPiece) {
        ctx.strokeStyle = 'rgba(220, 53, 69, 0.8)';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(x, y, PIECE_RADIUS + 5, 0, Math.PI * 2);
        ctx.stroke();
      } else {
        ctx.fillStyle = 'rgba(74, 144, 226, 0.5)';
        ctx.beginPath();
        ctx.arc(x, y, 10, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // Draw selection highlight
    if (selectedSquare) {
      const { x, y } = fileRankToPixel(selectedSquare.file, selectedSquare.rank);
      ctx.strokeStyle = COLORS.selectedGlow;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(x, y, PIECE_RADIUS + 6, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Draw hover highlight
    if (hoverSquare && !selectedSquare) {
      const piece = getPieceAt(hoverSquare.file, hoverSquare.rank);
      if (piece) {
        const { x, y } = fileRankToPixel(hoverSquare.file, hoverSquare.rank);
        ctx.strokeStyle = COLORS.hoverGlow;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, PIECE_RADIUS + 4, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    // Draw check indicator
    if (inCheckSquare) {
      const { x, y } = fileRankToPixel(inCheckSquare.file, inCheckSquare.rank);
      const time = Date.now() / 300;
      const pulse = Math.sin(time) * 0.4 + 0.6;
      const glowSize = PIECE_RADIUS + 8 + Math.sin(time) * 4;
      const gradient = ctx.createRadialGradient(x, y, PIECE_RADIUS - 5, x, y, glowSize + 15);
      gradient.addColorStop(0, 'rgba(255, 0, 0, 0)');
      gradient.addColorStop(0.5, `rgba(255, 0, 0, ${0.3 * pulse})`);
      gradient.addColorStop(1, 'rgba(255, 0, 0, 0)');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(x, y, glowSize + 15, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = `rgba(255, 0, 0, ${pulse})`;
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.arc(x, y, glowSize, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Draw pieces
    if (boardData.length > 0) {
      for (let rankIdx = 0; rankIdx < 10; rankIdx++) {
        for (let fileIdx = 0; fileIdx < 9; fileIdx++) {
          const piece = boardData[rankIdx]?.[fileIdx];
          if (!piece) continue;

          const rank = 9 - rankIdx;
          const { x, y } = fileRankToPixel(fileIdx, rank);
          const isHan = piece[0] === 'h';
          const pieceColor = isHan ? COLORS.hanColor : COLORS.choColor;
          const korean = boardKorean[rankIdx]?.[fileIdx];

          // Draw piece background
          const gradient = ctx.createRadialGradient(
            x - PIECE_RADIUS * 0.3,
            y - PIECE_RADIUS * 0.3,
            0,
            x,
            y,
            PIECE_RADIUS
          );
          gradient.addColorStop(0, '#fffef8');
          gradient.addColorStop(0.7, '#f5f0e0');
          gradient.addColorStop(1, '#e8e0d0');
          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.arc(x, y, PIECE_RADIUS, 0, Math.PI * 2);
          ctx.fill();

          // Draw piece border
          ctx.strokeStyle = pieceColor;
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.arc(x, y, PIECE_RADIUS - 1, 0, Math.PI * 2);
          ctx.stroke();

          // Draw piece character
          if (korean && korean.name) {
            ctx.fillStyle = pieceColor;
            ctx.font = 'bold 22px "Noto Serif KR", serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(korean.name, x, y + 1);
          }
        }
      }
    }
  };

  const initGame = async (formation: string | null = null, hanFormation: string | null = null, choFormation: string | null = null) => {
    try {
      const body: any = {
        game_id: GAME_ID,
        depth: 3,
        use_nnue: true,
      };

      if (formation) {
        body.formation = formation;
      } else if (hanFormation || choFormation) {
        if (hanFormation) body.han_formation = hanFormation;
        if (choFormation) body.cho_formation = choFormation;
      }

      const response = await fetch(`${API_BASE}/new-game`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (response.ok) {
        setSelectedSquare(null);
        setLastMoveFrom(null);
        setLastMoveTo(null);
        setIsGameOver(false);
        setIsAutoPlaying(false);
        await updateBoard();
      }
    } catch (error) {
      console.error('Failed to initialize game:', error);
      showToast('게임 초기화 실패', 'error');
    }
  };

  const updateBoard = async () => {
    try {
      const response = await fetch(`${API_BASE}/board/${GAME_ID}`);
      const data: BoardData = await response.json();

      setBoardData(data.board || []);
      setBoardKorean(data.board_korean || []);
      setLegalMoves(data.legal_moves || []);
      setIsGameOver(data.game_over || false);
      setCanUndo(data.can_undo || false);
      setInOpeningBook(data.in_opening_book || false);
      setMoveHistory(data.move_history || []);

      // Update status
      if (data.game_over) {
        if (data.winner) {
          const winnerName = data.winner === 'HAN' ? '한' : '초';
          setStatus(`게임 종료! ${winnerName} 측 승리!`);
          setStatusClass('status success');
        } else {
          if (data.draw_reason === 'repetition') {
            setStatus('게임 종료! 무승부 (동일 국면 3회 반복)');
          } else {
            setStatus('게임 종료! 무승부');
          }
          setStatusClass('status warning');
        }
      } else if (data.in_check) {
        const sideName = data.side_to_move === 'HAN' ? '한' : '초';
        setStatus(`${sideName} 측이 장군 상태입니다!`);
        setStatusClass('status warning');
      } else {
        const sideName = data.side_to_move === 'HAN' ? '한' : '초';
        setStatus(`현재 차례: ${sideName}`);
        setStatusClass('status info');
      }

      // Find king in check
      let checkSquare: Square | null = null;
      if (data.in_check) {
        const kingChar = data.side_to_move === 'HAN' ? 'hK' : 'cK';
        for (let rankIdx = 0; rankIdx < 10; rankIdx++) {
          for (let fileIdx = 0; fileIdx < 9; fileIdx++) {
            if (data.board[rankIdx]?.[fileIdx] === kingChar) {
              checkSquare = { file: fileIdx, rank: 9 - rankIdx };
              break;
            }
          }
          if (checkSquare) break;
        }
      }
      setInCheckSquare(checkSquare);
    } catch (error) {
      console.error('Failed to update board:', error);
    }
  };

  const selectSquare = (file: number, rank: number) => {
    const piece = getPieceAt(file, rank);
    if (!piece) return;

    setSelectedSquare({ file, rank });
    const squareKey = squareToNotation(file, rank);
    const movesFromSquare = legalMoves.filter((m) => m.from === squareKey);
    setPossibleMoveSquares(
      movesFromSquare.map((m) => {
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
          ? movesFromSquare.map((m) => m.to).join(', ')
          : '없음'
      );
    }
  };

  const clearSelection = () => {
    setSelectedSquare(null);
    setPossibleMoveSquares([]);
    setShowMoveInfo(false);
  };

  const squareToNotation = (file: number, rank: number) => {
    return String.fromCharCode(97 + file) + (rank + 1);
  };

  const makeMove = async (fromFile: number, fromRank: number, toFile: number, toRank: number) => {
    const fromSquare = squareToNotation(fromFile, fromRank);
    const toSquare = squareToNotation(toFile, toRank);

    try {
      const moveResponse = await fetch(`${API_BASE}/move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          game_id: GAME_ID,
          from_square: fromSquare,
          to_square: toSquare,
        }),
      });

      if (moveResponse.ok) {
        const result = await moveResponse.json();
        setLastMoveFrom({ file: fromFile, rank: fromRank });
        setLastMoveTo({ file: toFile, rank: toRank });
        clearSelection();
        await updateBoard();

        if (result.game_over && result.winner === null) {
          if (result.reason === 'draw_by_repetition') {
            showToast('동일 국면 3회 반복으로 무승부입니다!', 'warning', 5000);
          } else {
            showToast('무승부입니다!', 'warning', 5000);
          }
        }
      } else {
        const error = await moveResponse.json();
        showToast(`불법수: ${error.detail}`, 'error');
        clearSelection();
      }
    } catch (error) {
      console.error('Failed to make move:', error);
      clearSelection();
    }
  };

  const aiMove = async () => {
    if (isGameOver) {
      showToast('게임이 종료되었습니다. 새 게임을 시작하세요.', 'warning');
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/ai-move/${GAME_ID}`, {
        method: 'POST',
      });

      if (response.ok) {
        const data = await response.json();
        const fromFile = data.move.from.charCodeAt(0) - 97;
        const fromRank = parseInt(data.move.from.slice(1)) - 1;
        const toFile = data.move.to.charCodeAt(0) - 97;
        const toRank = parseInt(data.move.to.slice(1)) - 1;

        setLastMoveFrom({ file: fromFile, rank: fromRank });
        setLastMoveTo({ file: toFile, rank: toRank });
        await updateBoard();

        if (data.game_over && data.winner === null) {
          if (data.reason === 'draw_by_repetition') {
            showToast('동일 국면 3회 반복으로 무승부입니다!', 'warning', 5000);
          } else {
            showToast('무승부입니다!', 'warning', 5000);
          }
        } else {
          const bookInfo = data.from_opening_book ? ' 📖' : '';
          setStatus(`AI: ${data.move.from} → ${data.move.to} (${data.nodes_searched} nodes)${bookInfo}`);
          setStatusClass('status info');
        }

        if (isAutoPlaying && !data.game_over) {
          setTimeout(() => {
            if (isAutoPlaying && !isGameOver) {
              aiMove();
            }
          }, 500);
        }

        if (data.game_over) {
          setIsAutoPlaying(false);
        }
      } else {
        const error = await response.json();
        showToast(`AI 이동 실패: ${error.detail}`, 'error');
        setIsAutoPlaying(false);
      }
    } catch (error) {
      console.error('Failed to get AI move:', error);
      showToast('AI 이동 중 오류 발생', 'error');
      setIsAutoPlaying(false);
    }
  };

  const toggleAutoPlay = async () => {
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
      await aiMove();
    }
  };

  const undoMovePair = async () => {
    if (!canUndo || isAutoPlaying) {
      showToast('되돌릴 수 없습니다.', 'warning');
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/undo-pair/${GAME_ID}`, {
        method: 'POST',
      });

      if (response.ok) {
        clearSelection();
        setLastMoveFrom(null);
        setLastMoveTo(null);
        await updateBoard();
        showToast('두 수를 되돌렸습니다.', 'success');
      } else {
        await undoMove();
      }
    } catch (error) {
      console.error('Failed to undo move pair:', error);
      await undoMove();
    }
  };

  const undoMove = async () => {
    try {
      const response = await fetch(`${API_BASE}/undo/${GAME_ID}`, {
        method: 'POST',
      });

      if (response.ok) {
        clearSelection();
        setLastMoveFrom(null);
        setLastMoveTo(null);
        await updateBoard();
        showToast('수를 되돌렸습니다.', 'success');
      }
    } catch (error) {
      console.error('Failed to undo move:', error);
    }
  };

  const applyFormation = async () => {
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
    showToast(`${mySideLabel}: ${myFormation}, ${opponentSideLabel}: ${opponentFormation} 상차림으로 게임을 시작합니다!`, 'success');
  };

  const showToast = (message: string, type: string = 'info', duration: number = 3000) => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, duration);
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isGameOver) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const px = (e.clientX - rect.left) * scaleX;
    const py = (e.clientY - rect.top) * scaleY;

    const pos = pixelToFileRank(px, py);
    if (!pos) return;

    const { file, rank } = pos;

    if (selectedSquare) {
      const isPossibleMove = possibleMoveSquares.some(
        (sq) => sq.file === file && sq.rank === rank
      );

      if (isPossibleMove) {
        makeMove(selectedSquare.file, selectedSquare.rank, file, rank);
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
  };

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const px = (e.clientX - rect.left) * scaleX;
    const py = (e.clientY - rect.top) * scaleY;

    const pos = pixelToFileRank(px, py);
    if (pos) {
      setHoverSquare({ file: pos.file, rank: pos.rank });
    } else {
      setHoverSquare(null);
    }
  };

  const handleCanvasMouseLeave = () => {
    setHoverSquare(null);
  };

  return (
    <div className="container">
      <h1>장기 AI 엔진</h1>

      <div className="controls">
        <button onClick={() => initGame()}>새 게임</button>
        <button onClick={aiMove} disabled={isGameOver}>AI 이동</button>
        <button
          className={isAutoPlaying ? 'error' : 'secondary'}
          onClick={toggleAutoPlay}
          disabled={isGameOver}
        >
          {isAutoPlaying ? '자동 진행 중지' : 'AI 자동 진행'}
        </button>
        <button className="secondary" onClick={undoMovePair} disabled={!canUndo || isAutoPlaying}>
          되돌리기
        </button>
        <button className="secondary" onClick={() => setShowSetupPanel(!showSetupPanel)}>
          상차림 설정
        </button>
        <button className="secondary" onClick={() => setBoardFlipped(!boardFlipped)}>
          {boardFlipped ? '보드 되돌리기' : '보드 뒤집기'}
        </button>
        <Link to="/multiplayer" className="multiplayer-btn">
          🎮 멀티플레이어
        </Link>
      </div>

      <div className="game-info">
        <div className={statusClass}>{status}</div>
        <span
          className={`book-indicator ${inOpeningBook ? '' : 'inactive'}`}
          style={{ display: boardData.length > 0 ? 'inline-flex' : 'none' }}
        >
          📖 {inOpeningBook ? '정석북' : '정석북 종료'}
        </span>
      </div>

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
          <div className="side-panel" id="setupPanel">
            <h3>상차림 설정</h3>
            <div style={{ marginBottom: '24px' }}>
              <h4 style={{ color: '#c9a66b', marginBottom: '12px', fontSize: '16px', fontWeight: 600 }}>
                내 진영 선택
              </h4>
              <div className="formation-selector" style={{ display: 'flex', gap: '12px' }}>
                <div
                  className={`formation-option ${mySide === 'cho' ? 'selected' : ''}`}
                  style={{ flex: 1, textAlign: 'center', padding: '16px' }}
                  onClick={() => setMySide('cho')}
                >
                  <strong>초</strong>
                  <small>내가 초 진영</small>
                </div>
                <div
                  className={`formation-option ${mySide === 'han' ? 'selected' : ''}`}
                  style={{ flex: 1, textAlign: 'center', padding: '16px' }}
                  onClick={() => setMySide('han')}
                >
                  <strong>한</strong>
                  <small>내가 한 진영</small>
                </div>
              </div>
            </div>

            <div style={{ marginBottom: '24px' }}>
              <h4 style={{ color: '#c9a66b', marginBottom: '12px', fontSize: '16px', fontWeight: 600 }}>
                나의 상차림 ({mySide === 'cho' ? '초' : '한'} 측)
              </h4>
              <div className="formation-selector">
                {['상마상마', '마상마상', '마상상마', '상마마상'].map((formation) => (
                  <div
                    key={formation}
                    className={`formation-option ${myFormation === formation ? 'selected' : ''}`}
                    onClick={() => setMyFormation(formation)}
                  >
                    <strong>{formation}</strong>
                    <small>
                      b{mySide === 'cho' ? '10' : '1'}=
                      {formation[0] === '상' ? '상' : '마'}, c{mySide === 'cho' ? '10' : '1'}=
                      {formation[1] === '상' ? '상' : '마'}, g{mySide === 'cho' ? '10' : '1'}=
                      {formation[2] === '상' ? '상' : '마'}, h{mySide === 'cho' ? '10' : '1'}=
                      {formation[3] === '상' ? '상' : '마'}
                      {formation === '마상상마' ? ' (기본값)' : ''}
                    </small>
                  </div>
                ))}
              </div>
            </div>

            <div style={{ marginBottom: '24px' }}>
              <h4 style={{ color: '#c9a66b', marginBottom: '12px', fontSize: '16px', fontWeight: 600 }}>
                상대의 상차림 ({mySide === 'cho' ? '한' : '초'} 측)
              </h4>
              <div className="formation-selector">
                {['상마상마', '마상마상', '마상상마', '상마마상'].map((formation) => (
                  <div
                    key={formation}
                    className={`formation-option ${opponentFormation === formation ? 'selected' : ''}`}
                    onClick={() => setOpponentFormation(formation)}
                  >
                    <strong>{formation}</strong>
                    <small>
                      b{mySide === 'cho' ? '1' : '10'}=
                      {formation[0] === '상' ? '상' : '마'}, c{mySide === 'cho' ? '1' : '10'}=
                      {formation[1] === '상' ? '상' : '마'}, g{mySide === 'cho' ? '1' : '10'}=
                      {formation[2] === '상' ? '상' : '마'}, h{mySide === 'cho' ? '1' : '10'}=
                      {formation[3] === '상' ? '상' : '마'}
                      {formation === '마상상마' ? ' (기본값)' : ''}
                    </small>
                  </div>
                ))}
              </div>
            </div>

            <button onClick={applyFormation} style={{ width: '100%' }}>
              상차림 적용
            </button>
          </div>
        )}

        <div className="side-panel history-panel">
          <h3>이동 기록</h3>
          <div id="moveHistory">
            {moveHistory.length === 0 ? (
              <div style={{ color: '#6a6a7a', textAlign: 'center', padding: '24px', fontStyle: 'italic' }}>
                아직 이동이 없습니다
              </div>
            ) : (
              moveHistory.map((move) => (
                <div
                  key={move.move_number}
                  className={`history-item ${move.captured ? 'capture' : ''}`}
                >
                  <strong>{move.move_number}.</strong> {move.notation}
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <div className="toast-container">
        {toasts.map((toast) => (
          <div key={toast.id} className={`toast ${toast.type}`}>
            <span className="toast-icon">
              {toast.type === 'error' ? '⚠️' : toast.type === 'success' ? '✓' : toast.type === 'warning' ? '⚠️' : 'ℹ️'}
            </span>
            <span className="toast-content">{toast.message}</span>
            <button className="toast-close" onClick={() => setToasts((prev) => prev.filter((t) => t.id !== toast.id))}>
              ×
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default GamePage;

