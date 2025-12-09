interface GameStatusProps {
  status: string;
  statusClass: string;
  inOpeningBook: boolean;
  hasBoard: boolean;
}

export const GameStatus = ({ status, statusClass, inOpeningBook, hasBoard }: GameStatusProps) => {
  return (
    <div className="game-info">
      <div className={statusClass}>{status}</div>
      <span
        className={`book-indicator ${inOpeningBook ? '' : 'inactive'}`}
        style={{ display: hasBoard ? 'inline-flex' : 'none' }}
      >
        📖 {inOpeningBook ? '정석북' : '정석북 종료'}
      </span>
    </div>
  );
};

