export const API_BASE = '/api';
export const GAME_ID = 'default';

export const BASE_CANVAS_WIDTH = 520;
export const BASE_CANVAS_HEIGHT = 580;
export const PADDING = 40;
export const PIECE_RADIUS = 24;

export const COLORS = {
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
} as const;

export const FORMATIONS: Array<'상마상마' | '마상마상' | '마상상마' | '상마마상'> = [
  '상마상마',
  '마상마상',
  '마상상마',
  '상마마상',
];

export const DEFAULT_FORMATION = '마상상마';

