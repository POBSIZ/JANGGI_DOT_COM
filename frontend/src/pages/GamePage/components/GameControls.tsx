interface GameControlsProps {
  onNewGame: () => void;
  onAiMove: () => void;
  onToggleAutoPlay: () => void;
  onUndo: () => void;
  onToggleSetup: () => void;
  onToggleFlip: () => void;
  isGameOver: boolean;
  isAutoPlaying: boolean;
  canUndo: boolean;
  boardFlipped: boolean;
}

export const GameControls = ({
  onNewGame,
  onAiMove,
  onToggleAutoPlay,
  onUndo,
  onToggleSetup,
  onToggleFlip,
  isGameOver,
  isAutoPlaying,
  canUndo,
  boardFlipped,
}: GameControlsProps) => {
  return (
    <div className="controls">
      <button onClick={onNewGame}>새 게임</button>
      <button onClick={onAiMove} disabled={isGameOver}>
        AI 이동
      </button>
      <button
        className={isAutoPlaying ? 'error' : 'secondary'}
        onClick={onToggleAutoPlay}
        disabled={isGameOver}
      >
        {isAutoPlaying ? '자동 진행 중지' : 'AI 자동 진행'}
      </button>
      <button className="secondary" onClick={onUndo} disabled={!canUndo || isAutoPlaying}>
        되돌리기
      </button>
      <button className="secondary" onClick={onToggleSetup}>
        상차림 설정
      </button>
      <button className="secondary" onClick={onToggleFlip}>
        {boardFlipped ? '보드 되돌리기' : '보드 뒤집기'}
      </button>
    </div>
  );
};

