import { useEffect } from 'react';
import type { Square } from '../types';
import { PADDING, PIECE_RADIUS, COLORS } from '../constants';

interface BoardCanvasProps {
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  canvasWidth: number;
  canvasHeight: number;
  boardData: string[][];
  boardKorean: Array<Array<{ name: string; full_name?: string }>>;
  selectedSquare: Square | null;
  possibleMoveSquares: Square[];
  lastMoveFrom: Square | null;
  lastMoveTo: Square | null;
  inCheckSquare: Square | null;
  hoverSquare: Square | null;
  boardFlipped: boolean;
  fileRankToPixel: (file: number, rank: number) => { x: number; y: number };
  getPieceAt: (file: number, rank: number) => string | null;
}

export const BoardCanvas = ({
  canvasRef,
  canvasWidth,
  canvasHeight,
  boardData,
  boardKorean,
  selectedSquare,
  possibleMoveSquares,
  lastMoveFrom,
  lastMoveTo,
  inCheckSquare,
  hoverSquare,
  boardFlipped,
  fileRankToPixel,
  getPieceAt,
}: BoardCanvasProps) => {
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || boardData.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.fillStyle = COLORS.boardBg;
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    ctx.strokeStyle = COLORS.boardLine;
    ctx.lineWidth = 1.5;

    const CELL_WIDTH = (canvasWidth - PADDING * 2) / 8;
    const CELL_HEIGHT = (canvasHeight - PADDING * 2) / 9;

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
  }, [
    canvasRef,
    canvasWidth,
    canvasHeight,
    boardData,
    boardKorean,
    selectedSquare,
    possibleMoveSquares,
    lastMoveFrom,
    lastMoveTo,
    inCheckSquare,
    hoverSquare,
    boardFlipped,
    fileRankToPixel,
    getPieceAt,
  ]);

  // This component doesn't render anything, it only handles canvas drawing via useEffect
  return null;
};

