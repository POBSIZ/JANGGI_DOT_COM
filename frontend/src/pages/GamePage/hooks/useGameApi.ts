import { useState, useCallback } from 'react';
import type { BoardData, Square } from '../types';
import { API_BASE, GAME_ID } from '../constants';
import { squareToNotation, notationToSquare } from '../utils';

export const useGameApi = (showToast: (message: string, type?: 'info' | 'success' | 'warning' | 'error', duration?: number) => void) => {
  const [boardData, setBoardData] = useState<string[][]>([]);
  const [boardKorean, setBoardKorean] = useState<Array<Array<{ name: string; full_name?: string }>>>([]);
  const [legalMoves, setLegalMoves] = useState<Array<{ from: string; to: string }>>([]);
  const [isGameOver, setIsGameOver] = useState(false);
  const [canUndo, setCanUndo] = useState(false);
  const [inOpeningBook, setInOpeningBook] = useState(false);
  const [moveHistory, setMoveHistory] = useState<Array<{ move_number: number; notation: string; captured?: boolean }>>([]);
  const [status, setStatus] = useState('게임을 시작하세요');
  const [statusClass, setStatusClass] = useState('status info');
  const [inCheckSquare, setInCheckSquare] = useState<Square | null>(null);
  const [lastMoveFrom, setLastMoveFrom] = useState<Square | null>(null);
  const [lastMoveTo, setLastMoveTo] = useState<Square | null>(null);

  const updateBoard = useCallback(async () => {
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
  }, []);

  const initGame = useCallback(
    async (formation: string | null = null, hanFormation: string | null = null, choFormation: string | null = null) => {
      try {
        const body: {
          game_id: string;
          depth: number;
          use_nnue: boolean;
          formation?: string;
          han_formation?: string;
          cho_formation?: string;
        } = {
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
          setLastMoveFrom(null);
          setLastMoveTo(null);
          setIsGameOver(false);
          await updateBoard();
        }
      } catch (error) {
        console.error('Failed to initialize game:', error);
        showToast('게임 초기화 실패', 'error');
      }
    },
    [updateBoard, showToast]
  );

  const makeMove = useCallback(
    async (fromFile: number, fromRank: number, toFile: number, toRank: number) => {
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
          await updateBoard();

          if (result.game_over && result.winner === null) {
            if (result.reason === 'draw_by_repetition') {
              showToast('동일 국면 3회 반복으로 무승부입니다!', 'warning', 5000);
            } else {
              showToast('무승부입니다!', 'warning', 5000);
            }
          }
          return true;
        } else {
          const error = await moveResponse.json();
          showToast(`불법수: ${error.detail}`, 'error');
          return false;
        }
      } catch (error) {
        console.error('Failed to make move:', error);
        return false;
      }
    },
    [updateBoard, showToast]
  );

  const aiMove = useCallback(
    async () => {
      if (isGameOver) {
        showToast('게임이 종료되었습니다. 새 게임을 시작하세요.', 'warning');
        return false;
      }

      try {
        const response = await fetch(`${API_BASE}/ai-move/${GAME_ID}`, {
          method: 'POST',
        });

        if (response.ok) {
          const data = await response.json();
          const fromSquare = notationToSquare(data.move.from);
          const toSquare = notationToSquare(data.move.to);

          setLastMoveFrom(fromSquare);
          setLastMoveTo(toSquare);
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

          return { gameOver: data.game_over || false };
        } else {
          const error = await response.json();
          showToast(`AI 이동 실패: ${error.detail}`, 'error');
          return { gameOver: true };
        }
      } catch (error) {
        console.error('Failed to get AI move:', error);
        showToast('AI 이동 중 오류 발생', 'error');
        return { gameOver: true };
      }
    },
    [isGameOver, updateBoard, showToast]
  );

  const undoMove = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/undo/${GAME_ID}`, {
        method: 'POST',
      });

      if (response.ok) {
        setLastMoveFrom(null);
        setLastMoveTo(null);
        await updateBoard();
        showToast('수를 되돌렸습니다.', 'success');
      }
    } catch (error) {
      console.error('Failed to undo move:', error);
    }
  }, [updateBoard, showToast]);

  const undoMovePair = useCallback(async () => {
    if (!canUndo) {
      showToast('되돌릴 수 없습니다.', 'warning');
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/undo-pair/${GAME_ID}`, {
        method: 'POST',
      });

      if (response.ok) {
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
  }, [canUndo, updateBoard, showToast, undoMove]);

  return {
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
    updateBoard,
    initGame,
    makeMove,
    aiMove,
    undoMovePair,
    undoMove,
  };
};

