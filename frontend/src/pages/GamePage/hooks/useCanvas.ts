import { useState, useEffect, useRef, useCallback } from 'react';
import type { Square } from '../types';
import { BASE_CANVAS_WIDTH, BASE_CANVAS_HEIGHT, PADDING } from '../constants';

export const useCanvas = (boardFlipped: boolean) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [canvasWidth, setCanvasWidth] = useState(BASE_CANVAS_WIDTH);
  const [canvasHeight, setCanvasHeight] = useState(BASE_CANVAS_HEIGHT);

  const CELL_WIDTH = (canvasWidth - PADDING * 2) / 8;
  const CELL_HEIGHT = (canvasHeight - PADDING * 2) / 9;

  const resizeCanvas = useCallback(() => {
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
  }, []);

  useEffect(() => {
    // Initial resize using requestAnimationFrame to avoid setState in effect
    const initialResize = () => {
      const container = document.querySelector('.board-wrapper');
      if (container) {
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
      }
    };

    requestAnimationFrame(initialResize);
    window.addEventListener('resize', resizeCanvas);
    return () => window.removeEventListener('resize', resizeCanvas);
  }, [resizeCanvas]);

  const fileRankToPixel = useCallback(
    (file: number, rank: number) => {
      const displayFile = boardFlipped ? 8 - file : file;
      const displayRank = boardFlipped ? 9 - rank : rank;
      const x = PADDING + displayFile * CELL_WIDTH;
      const y = PADDING + (9 - displayRank) * CELL_HEIGHT;
      return { x, y };
    },
    [boardFlipped, CELL_WIDTH, CELL_HEIGHT]
  );

  const pixelToFileRank = useCallback(
    (px: number, py: number): Square | null => {
      const displayFile = Math.round((px - PADDING) / CELL_WIDTH);
      const rankFromTop = Math.round((py - PADDING) / CELL_HEIGHT);
      const displayRank = 9 - rankFromTop;
      const file = boardFlipped ? 8 - displayFile : displayFile;
      const rank = boardFlipped ? 9 - displayRank : displayRank;

      if (file >= 0 && file <= 8 && rank >= 0 && rank <= 9) {
        return { file, rank };
      }
      return null;
    },
    [boardFlipped, CELL_WIDTH, CELL_HEIGHT]
  );

  const getCanvasCoordinates = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const px = (e.clientX - rect.left) * scaleX;
    const py = (e.clientY - rect.top) * scaleY;

    return pixelToFileRank(px, py);
  }, [pixelToFileRank]);

  return {
    canvasRef,
    canvasWidth,
    canvasHeight,
    CELL_WIDTH,
    CELL_HEIGHT,
    fileRankToPixel,
    pixelToFileRank,
    getCanvasCoordinates,
  };
};

