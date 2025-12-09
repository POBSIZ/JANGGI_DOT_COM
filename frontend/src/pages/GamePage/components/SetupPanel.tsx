import type { Side, Formation } from '../types';
import { FORMATIONS, DEFAULT_FORMATION } from '../constants';

interface SetupPanelProps {
  mySide: Side;
  myFormation: Formation;
  opponentFormation: Formation;
  onMySideChange: (side: Side) => void;
  onMyFormationChange: (formation: Formation) => void;
  onOpponentFormationChange: (formation: Formation) => void;
  onApply: () => void;
}

export const SetupPanel = ({
  mySide,
  myFormation,
  opponentFormation,
  onMySideChange,
  onMyFormationChange,
  onOpponentFormationChange,
  onApply,
}: SetupPanelProps) => {
  return (
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
            onClick={() => onMySideChange('cho')}
          >
            <strong>초</strong>
            <small>내가 초 진영</small>
          </div>
          <div
            className={`formation-option ${mySide === 'han' ? 'selected' : ''}`}
            style={{ flex: 1, textAlign: 'center', padding: '16px' }}
            onClick={() => onMySideChange('han')}
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
          {FORMATIONS.map((formation) => (
            <div
              key={formation}
              className={`formation-option ${myFormation === formation ? 'selected' : ''}`}
              onClick={() => onMyFormationChange(formation)}
            >
              <strong>{formation}</strong>
              <small>
                b{mySide === 'cho' ? '10' : '1'}=
                {formation[0] === '상' ? '상' : '마'}, c{mySide === 'cho' ? '10' : '1'}=
                {formation[1] === '상' ? '상' : '마'}, g{mySide === 'cho' ? '10' : '1'}=
                {formation[2] === '상' ? '상' : '마'}, h{mySide === 'cho' ? '10' : '1'}=
                {formation[3] === '상' ? '상' : '마'}
                {formation === DEFAULT_FORMATION ? ' (기본값)' : ''}
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
          {FORMATIONS.map((formation) => (
            <div
              key={formation}
              className={`formation-option ${opponentFormation === formation ? 'selected' : ''}`}
              onClick={() => onOpponentFormationChange(formation)}
            >
              <strong>{formation}</strong>
              <small>
                b{mySide === 'cho' ? '1' : '10'}=
                {formation[0] === '상' ? '상' : '마'}, c{mySide === 'cho' ? '1' : '10'}=
                {formation[1] === '상' ? '상' : '마'}, g{mySide === 'cho' ? '1' : '10'}=
                {formation[2] === '상' ? '상' : '마'}, h{mySide === 'cho' ? '1' : '10'}=
                {formation[3] === '상' ? '상' : '마'}
                {formation === DEFAULT_FORMATION ? ' (기본값)' : ''}
              </small>
            </div>
          ))}
        </div>
      </div>

      <button onClick={onApply} style={{ width: '100%' }}>
        상차림 적용
      </button>
    </div>
  );
};

