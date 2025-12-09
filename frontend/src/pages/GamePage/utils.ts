import type { Square } from './types';

export const squareToNotation = (file: number, rank: number): string => {
  return String.fromCharCode(97 + file) + (rank + 1);
};

export const notationToSquare = (notation: string): Square => {
  const file = notation.charCodeAt(0) - 97;
  const rank = parseInt(notation.slice(1)) - 1;
  return { file, rank };
};

export const flipFormation = (formation: string): string => {
  const flipMap: Record<string, string> = {
    '상마상마': '마상마상',
    '마상마상': '상마상마',
    '마상상마': '마상상마',
    '상마마상': '상마마상',
  };
  return flipMap[formation] || formation;
};

export const getSideName = (side: 'HAN' | 'CHO'): string => {
  return side === 'HAN' ? '한' : '초';
};

