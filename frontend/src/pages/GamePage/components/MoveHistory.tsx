import type { MoveHistoryItem } from '../types';

interface MoveHistoryProps {
  moveHistory: MoveHistoryItem[];
}

export const MoveHistory = ({ moveHistory }: MoveHistoryProps) => {
  return (
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
  );
};

