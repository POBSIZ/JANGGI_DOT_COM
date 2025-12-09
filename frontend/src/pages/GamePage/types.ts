export interface BoardData {
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

export interface Square {
  file: number;
  rank: number;
}

export interface Toast {
  id: number;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
}

export interface MoveHistoryItem {
  move_number: number;
  notation: string;
  captured?: boolean;
}

export type Side = 'cho' | 'han';
export type Formation = '상마상마' | '마상마상' | '마상상마' | '상마마상';

