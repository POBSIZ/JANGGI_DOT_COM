import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import './MultiplayerPage.css';

interface Room {
  room_id: string;
  players: Record<string, { player_id: string; nickname: string; side: 'CHO' | 'HAN'; is_ready: boolean; formation?: string }>;
  current_turn?: 'CHO' | 'HAN';
  time_limit?: number;
}

const MultiplayerPage = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [playerId, setPlayerId] = useState<string | null>(null);
  const [nickname, setNickname] = useState('');
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [currentRoom, setCurrentRoom] = useState<Room | null>(null);
  const [myPlayerSide, setMyPlayerSide] = useState<'CHO' | 'HAN' | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [gameInProgress, setGameInProgress] = useState(false);
  const [boardData, setBoardData] = useState<string[][]>([]);
  const [boardKorean, setBoardKorean] = useState<Array<Array<{ name: string; full_name?: string }>>>([]);
  const [selectedSquare, setSelectedSquare] = useState<{ file: number; rank: number } | null>(null);
  const [legalMoves, setLegalMoves] = useState<Array<{ from: string; to: string }>>([]);
  const [lastMoveFrom, setLastMoveFrom] = useState<{ file: number; rank: number } | null>(null);
  const [lastMoveTo, setLastMoveTo] = useState<{ file: number; rank: number } | null>(null);
  const [currentTurn, setCurrentTurn] = useState<'CHO' | 'HAN'>('CHO');
  const [boardFlipped, setBoardFlipped] = useState(false);
  const [showLobby, setShowLobby] = useState(true);
  const [rooms, setRooms] = useState<Room[]>([]);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showJoinModal, setShowJoinModal] = useState(false);
  const [createNickname, setCreateNickname] = useState('');
  const [joinNickname, setJoinNickname] = useState('');
  const [joinRoomCode, setJoinRoomCode] = useState('');
  const [timeLimit, setTimeLimit] = useState('');
  const [formation, setFormation] = useState('마상상마');
  const [chatMessages, setChatMessages] = useState<Array<{ player_id: string; nickname: string; message: string; type?: string }>>([]);
  const [chatInput, setChatInput] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [showDrawModal, setShowDrawModal] = useState(false);

  const CANVAS_WIDTH = 520;
  const CANVAS_HEIGHT = 580;
  const PADDING = 40;
  const CELL_WIDTH = (CANVAS_WIDTH - PADDING * 2) / 8;
  const CELL_HEIGHT = (CANVAS_HEIGHT - PADDING * 2) / 9;
  const PIECE_RADIUS = 24;

  const COLORS = {
    boardBg: '#c9a66b',
    boardLine: '#3d2914',
    hanColor: '#c41e3a',
    choColor: '#1a5fb4',
    selectedGlow: '#90ee90',
    possibleMove: 'rgba(74, 144, 226, 0.6)',
    lastMoveFrom: 'rgba(255, 215, 0, 0.4)',
    lastMoveTo: 'rgba(50, 205, 50, 0.4)',
  };

  useEffect(() => {
    initConnection();
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, []);

  useEffect(() => {
    if (canvasRef.current && boardData.length > 0) {
      drawBoard();
    }
  }, [boardData, selectedSquare, lastMoveFrom, lastMoveTo, boardFlipped]);

  const initConnection = async () => {
    try {
      const response = await fetch('/api/multiplayer/generate-player-id', { method: 'POST' });
      const data = await response.json();
      setPlayerId(data.player_id);

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsConnection = new WebSocket(`${protocol}//${window.location.host}/ws/multiplayer/${data.player_id}`);

      wsConnection.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        refreshRooms();
      };

      wsConnection.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        setTimeout(initConnection, 3000);
      };

      wsConnection.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      };

      wsConnection.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMessage(data);
      };

      setWs(wsConnection);

      setInterval(() => {
        if (wsConnection && wsConnection.readyState === WebSocket.OPEN) {
          wsConnection.send(JSON.stringify({ type: 'ping' }));
        }
      }, 30000);
    } catch (error) {
      console.error('Failed to initialize connection:', error);
    }
  };

  const sendMessage = (msg: any) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(msg));
    }
  };

  const handleMessage = (data: any) => {
    switch (data.type) {
      case 'room_created':
        setCurrentRoom(data.room);
        setMyPlayerSide(data.your_side);
        setShowLobby(false);
        showToast('방이 생성되었습니다!', 'success');
        addSystemMessage('방이 생성되었습니다. 상대방을 기다리는 중...');
        break;
      case 'room_joined':
        setCurrentRoom(data.room);
        setMyPlayerSide(data.your_side);
        setShowLobby(false);
        showToast('방에 입장했습니다!', 'success');
        addSystemMessage('방에 입장했습니다.');
        break;
      case 'player_joined':
        setCurrentRoom(data.room);
        showToast(`${data.player.nickname}님이 입장했습니다`, 'success');
        addSystemMessage(`${data.player.nickname}님이 입장했습니다.`);
        break;
      case 'player_left':
        setCurrentRoom(data.room);
        showToast('상대방이 나갔습니다', 'error');
        addSystemMessage('상대방이 나갔습니다.');
        if (gameInProgress) {
          setGameInProgress(false);
        }
        break;
      case 'room_left':
        setCurrentRoom(null);
        setMyPlayerSide(null);
        setIsReady(false);
        setGameInProgress(false);
        setShowLobby(true);
        refreshRooms();
        break;
      case 'formation_changed':
      case 'ready_changed':
        setCurrentRoom(data.room);
        break;
      case 'game_started':
        setCurrentRoom(data.room);
        setGameInProgress(true);
        setIsReady(false);
        initializeBoard(data.cho_formation, data.han_formation);
        addSystemMessage('게임이 시작되었습니다!');
        showToast('게임 시작!', 'success');
        break;
      case 'move_made':
        setCurrentRoom(data.room);
        const move = data.move;
        setLastMoveFrom(parseSquare(move.from));
        setLastMoveTo(parseSquare(move.to));
        setCurrentTurn(data.room.current_turn);
        updateBoardFromServer();
        break;
      case 'game_over':
        setGameInProgress(false);
        setCurrentRoom(data.room);
        let message = '';
        if (data.reason === 'timeout') {
          message = `시간 초과! ${data.winner === 'CHO' ? '초' : '한'} 측 승리!`;
        } else if (data.reason === 'resignation') {
          message = `기권! ${data.winner === 'CHO' ? '초' : '한'} 측 승리!`;
        } else if (data.reason === 'draw_agreed') {
          message = '무승부로 합의되었습니다.';
        } else if (data.reason === 'checkmate') {
          message = `체크메이트! ${data.winner === 'CHO' ? '초' : '한'} 측 승리!`;
        }
        addSystemMessage(message);
        showToast(message, data.winner === myPlayerSide ? 'success' : 'error');
        break;
      case 'chat':
        setChatMessages((prev) => [...prev, data]);
        break;
      case 'draw_offered':
        setShowDrawModal(true);
        break;
      case 'draw_declined':
        showToast('무승부 제안이 거절되었습니다', 'error');
        break;
      case 'room_list':
        setRooms(data.rooms || []);
        break;
      case 'error':
        showToast(data.message, 'error');
        break;
    }
  };

  const createRoom = () => {
    const nick = createNickname.trim() || `Player_${playerId?.substring(0, 6)}`;
    setNickname(nick);
    sendMessage({
      type: 'create_room',
      nickname: nick,
      time_limit: timeLimit ? parseInt(timeLimit) : null,
    });
    setShowCreateModal(false);
    setCreateNickname('');
    setTimeLimit('');
  };

  const joinRoomByCode = () => {
    const nick = joinNickname.trim() || `Player_${playerId?.substring(0, 6)}`;
    const roomCode = joinRoomCode.trim().toUpperCase();
    if (!roomCode) {
      showToast('방 코드를 입력해주세요', 'error');
      return;
    }
    setNickname(nick);
    sendMessage({
      type: 'join_room',
      nickname: nick,
      room_id: roomCode,
    });
    setShowJoinModal(false);
    setJoinNickname('');
    setJoinRoomCode('');
  };

  const quickJoinRoom = (roomId: string) => {
    const nick = prompt('닉네임을 입력하세요:', `Player_${playerId?.substring(0, 6)}`);
    if (!nick) return;
    setNickname(nick);
    sendMessage({
      type: 'join_room',
      nickname: nick,
      room_id: roomId,
    });
  };

  const refreshRooms = () => {
    sendMessage({ type: 'get_rooms' });
  };

  const leaveRoom = () => {
    if (gameInProgress && !confirm('게임 중입니다. 정말 나가시겠습니까?')) {
      return;
    }
    sendMessage({ type: 'leave_room' });
  };


  const toggleReady = () => {
    const newReady = !isReady;
    setIsReady(newReady);
    sendMessage({
      type: 'set_ready',
      is_ready: newReady,
    });
  };

  const sendChat = () => {
    if (!chatInput.trim()) return;
    sendMessage({
      type: 'chat',
      message: chatInput,
    });
    setChatInput('');
  };

  const handleChatKeypress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      sendChat();
    }
  };

  const offerDraw = () => {
    if (confirm('무승부를 제안하시겠습니까?')) {
      sendMessage({ type: 'offer_draw' });
      showToast('무승부를 제안했습니다', 'info');
    }
  };

  const respondDraw = (accept: boolean) => {
    sendMessage({
      type: 'respond_draw',
      accept: accept,
    });
    setShowDrawModal(false);
  };

  const resign = () => {
    if (confirm('정말 기권하시겠습니까?')) {
      sendMessage({ type: 'resign' });
    }
  };

  const fileRankToPixel = (file: number, rank: number) => {
    const displayFile = boardFlipped ? (8 - file) : file;
    const displayRank = boardFlipped ? (9 - rank) : rank;
    const x = PADDING + displayFile * CELL_WIDTH;
    const y = PADDING + (9 - displayRank) * CELL_HEIGHT;
    return { x, y };
  };

  const pixelToFileRank = (px: number, py: number): { file: number; rank: number } | null => {
    const displayFile = Math.round((px - PADDING) / CELL_WIDTH);
    const rankFromTop = Math.round((py - PADDING) / CELL_HEIGHT);
    const displayRank = 9 - rankFromTop;
    const file = boardFlipped ? (8 - displayFile) : displayFile;
    const rank = boardFlipped ? (9 - displayRank) : displayRank;

    if (file >= 0 && file <= 8 && rank >= 0 && rank <= 9) {
      return { file, rank };
    }
    return null;
  };

  const getPieceAt = (file: number, rank: number): string | null => {
    const rankIdx = 9 - rank;
    return boardData[rankIdx]?.[file] || null;
  };

  const isMyPiece = (piece: string | null): boolean => {
    if (!piece || !myPlayerSide) return false;
    const pieceSide = piece[0] === 'h' ? 'HAN' : 'CHO';
    return pieceSide === myPlayerSide;
  };

  const squareToNotation = (file: number, rank: number) => {
    return String.fromCharCode(97 + file) + (rank + 1);
  };

  const parseSquare = (notation: string) => {
    const file = notation.charCodeAt(0) - 97;
    const rank = parseInt(notation.slice(1)) - 1;
    return { file, rank };
  };

  const initializeBoard = (choFormation: string, hanFormation: string) => {
    const board = Array(10).fill(null).map(() => Array(9).fill(null));
    const formations: Record<string, string[]> = {
      '상마상마': ['E', 'H', 'E', 'H'],
      '마상마상': ['H', 'E', 'H', 'E'],
      '마상상마': ['H', 'E', 'E', 'H'],
      '상마마상': ['E', 'H', 'H', 'E'],
    };

    const hanF = formations[hanFormation] || formations['마상상마'];
    const choF = formations[choFormation] || formations['마상상마'];

    board[0][0] = 'hR'; board[0][1] = 'h' + hanF[0]; board[0][2] = 'h' + hanF[1];
    board[0][3] = 'hG'; board[0][5] = 'hG';
    board[0][6] = 'h' + hanF[2]; board[0][7] = 'h' + hanF[3]; board[0][8] = 'hR';
    board[1][4] = 'hK';
    board[2][1] = 'hC'; board[2][7] = 'hC';
    board[3][0] = 'hP'; board[3][2] = 'hP'; board[3][4] = 'hP';
    board[3][6] = 'hP'; board[3][8] = 'hP';

    board[9][0] = 'cR'; board[9][1] = 'c' + choF[0]; board[9][2] = 'c' + choF[1];
    board[9][3] = 'cG'; board[9][5] = 'cG';
    board[9][6] = 'c' + choF[2]; board[9][7] = 'c' + choF[3]; board[9][8] = 'cR';
    board[8][4] = 'cK';
    board[7][1] = 'cC'; board[7][7] = 'cC';
    board[6][0] = 'cP'; board[6][2] = 'cP'; board[6][4] = 'cP';
    board[6][6] = 'cP'; board[6][8] = 'cP';

    setBoardData(board);
    setCurrentTurn('CHO');
    setSelectedSquare(null);
    setLastMoveFrom(null);
    setLastMoveTo(null);
    setBoardFlipped(myPlayerSide === 'HAN');
  };

  const updateBoardFromServer = async () => {
    if (!currentRoom) return;
    try {
      const response = await fetch(`/api/board/mp_${currentRoom.room_id}`);
      if (response.ok) {
        const data = await response.json();
        setBoardData(data.board);
        setBoardKorean(data.board_korean);
        setLegalMoves(data.legal_moves || []);
      }
    } catch (e) {
      console.error('Failed to update board:', e);
    }
  };

  const drawBoard = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = COLORS.boardBg;
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    ctx.strokeStyle = COLORS.boardLine;
    ctx.lineWidth = 1.5;

    for (let i = 0; i < 10; i++) {
      const y = PADDING + i * CELL_HEIGHT;
      ctx.beginPath();
      ctx.moveTo(PADDING, y);
      ctx.lineTo(CANVAS_WIDTH - PADDING, y);
      ctx.stroke();
    }

    for (let i = 0; i < 9; i++) {
      const x = PADDING + i * CELL_WIDTH;
      ctx.beginPath();
      ctx.moveTo(x, PADDING);
      ctx.lineTo(x, CANVAS_HEIGHT - PADDING);
      ctx.stroke();
    }

    // Draw palace diagonals
    const palaces = [
      { corners: [[3, 9], [5, 9], [3, 7], [5, 7]] },
      { corners: [[3, 2], [5, 2], [3, 0], [5, 0]] },
    ];

    palaces.forEach(palace => {
      const c = palace.corners;
      const tl = fileRankToPixel(c[0][0], c[0][1]);
      const tr = fileRankToPixel(c[1][0], c[1][1]);
      const bl = fileRankToPixel(c[2][0], c[2][1]);
      const br = fileRankToPixel(c[3][0], c[3][1]);

      ctx.beginPath();
      ctx.moveTo(tl.x, tl.y);
      ctx.lineTo(br.x, br.y);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(tr.x, tr.y);
      ctx.lineTo(bl.x, bl.y);
      ctx.stroke();
    });

    // Draw highlights
    if (lastMoveFrom) {
      const { x, y } = fileRankToPixel(lastMoveFrom.file, lastMoveFrom.rank);
      ctx.fillStyle = COLORS.lastMoveFrom;
      ctx.beginPath();
      ctx.arc(x, y, PIECE_RADIUS + 5, 0, Math.PI * 2);
      ctx.fill();
    }

    if (lastMoveTo) {
      const { x, y } = fileRankToPixel(lastMoveTo.file, lastMoveTo.rank);
      ctx.fillStyle = COLORS.lastMoveTo;
      ctx.beginPath();
      ctx.arc(x, y, PIECE_RADIUS + 5, 0, Math.PI * 2);
      ctx.fill();
    }

    if (selectedSquare) {
      const { x, y } = fileRankToPixel(selectedSquare.file, selectedSquare.rank);
      ctx.strokeStyle = COLORS.selectedGlow;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(x, y, PIECE_RADIUS + 4, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Draw pieces
    if (boardData.length > 0) {
      const pieceNames: Record<string, string> = {
        'K': '왕', 'G': '사', 'E': '상', 'H': '마',
        'R': '차', 'C': '포', 'P': '졸',
      };

      for (let rankIdx = 0; rankIdx < 10; rankIdx++) {
        for (let fileIdx = 0; fileIdx < 9; fileIdx++) {
          const piece = boardData[rankIdx]?.[fileIdx];
          if (!piece) continue;

          const rank = 9 - rankIdx;
          const { x, y } = fileRankToPixel(fileIdx, rank);
          const isHan = piece[0] === 'h';
          const pieceType = piece[1];
          const pieceName = pieceNames[pieceType] || '?';
          const pieceColor = isHan ? COLORS.hanColor : COLORS.choColor;

          ctx.fillStyle = '#faf6e8';
          ctx.beginPath();
          ctx.arc(x, y, PIECE_RADIUS, 0, Math.PI * 2);
          ctx.fill();

          ctx.strokeStyle = pieceColor;
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.arc(x, y, PIECE_RADIUS - 1, 0, Math.PI * 2);
          ctx.stroke();

          ctx.fillStyle = pieceColor;
          ctx.font = 'bold 22px "Noto Serif KR", serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(pieceName, x, y + 1);
        }
      }
    }
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!gameInProgress) return;
    if (currentTurn !== myPlayerSide) {
      showToast('상대방 차례입니다', 'error');
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    const pos = pixelToFileRank(px, py);
    if (!pos) return;

    const { file, rank } = pos;
    const piece = getPieceAt(file, rank);

    if (selectedSquare) {
      if (file !== selectedSquare.file || rank !== selectedSquare.rank) {
        makeMove(selectedSquare.file, selectedSquare.rank, file, rank);
      }
      setSelectedSquare(null);
    } else if (piece && isMyPiece(piece)) {
      setSelectedSquare({ file, rank });
    }
  };

  const makeMove = (fromFile: number, fromRank: number, toFile: number, toRank: number) => {
    const fromSquare = squareToNotation(fromFile, fromRank);
    const toSquare = squareToNotation(toFile, toRank);

    const rankIdxFrom = 9 - fromRank;
    const rankIdxTo = 9 - toRank;
    const piece = boardData[rankIdxFrom][fromFile];

    if (piece) {
      const newBoard = [...boardData];
      newBoard[rankIdxTo] = [...newBoard[rankIdxTo]];
      newBoard[rankIdxTo][toFile] = piece;
      newBoard[rankIdxFrom] = [...newBoard[rankIdxFrom]];
      newBoard[rankIdxFrom][fromFile] = null;
      setBoardData(newBoard);

      setLastMoveFrom({ file: fromFile, rank: fromRank });
      setLastMoveTo({ file: toFile, rank: toRank });
    }

    sendMessage({
      type: 'make_move',
      from_square: fromSquare,
      to_square: toSquare,
    });
  };

  const showToast = (message: string, type: string = 'info') => {
    // Simple toast implementation - you can enhance this
    console.log(`[${type.toUpperCase()}] ${message}`);
  };

  const addSystemMessage = (text: string) => {
    setChatMessages((prev) => [...prev, { player_id: 'system', nickname: '시스템', message: text, type: 'system' }]);
  };

  const getSideForPlayer = (pid: string) => {
    if (!currentRoom) return '';
    const player = currentRoom.players[pid];
    return player?.side?.toLowerCase() || '';
  };

  const escapeHtml = (text: string) => {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  };

  const players = currentRoom ? Object.values(currentRoom.players) : [];
  const choPlayer = players.find(p => p.side === 'CHO');
  const hanPlayer = players.find(p => p.side === 'HAN');

  return (
    <div>
      <header className="header">
        <h1>장기 멀티플레이어</h1>
        <div className="header-buttons">
          <div className="connection-status">
            <div className={`connection-dot ${isConnected ? '' : 'disconnected'}`}></div>
            <span>{isConnected ? '연결됨' : '연결 끊김'}</span>
          </div>
          <Link to="/" className="btn btn-secondary">AI 대전</Link>
        </div>
      </header>

      {showLobby ? (
        <div className="lobby-container">
          <div className="lobby-header">
            <h2>🎮 온라인 대전</h2>
            <p>다른 플레이어와 실시간으로 대국하세요</p>
          </div>

          <div className="lobby-actions">
            <button className="btn btn-primary" onClick={() => setShowCreateModal(true)}>
              ➕ 방 만들기
            </button>
            <button className="btn btn-secondary" onClick={() => setShowJoinModal(true)}>
              🚪 방 코드로 입장
            </button>
            <button className="btn btn-secondary" onClick={refreshRooms}>
              🔄 새로고침
            </button>
          </div>

          <div className="room-list">
            <h3>대기중인 방</h3>
            <div>
              {rooms.length === 0 ? (
                <div className="no-rooms">아직 대기중인 방이 없습니다</div>
              ) : (
                rooms.map(room => {
                  const hostPlayer = Object.values(room.players)[0];
                  return (
                    <div key={room.room_id} className="room-item">
                      <div className="room-info">
                        <div className="room-id">방 코드: {room.room_id}</div>
                        <div className="room-details">
                          호스트: {hostPlayer?.nickname || '알 수 없음'} |
                          시간: {room.time_limit ? (room.time_limit / 60) + '분' : '무제한'}
                        </div>
                      </div>
                      <button className="btn btn-primary" onClick={() => quickJoinRoom(room.room_id)}>
                        입장
                      </button>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="room-container">
          <div className="room-header">
            <div className="room-title">
              <h2>대국실</h2>
              <span className="room-code">{currentRoom?.room_id || '-'}</span>
            </div>
            <div>
              <button className="btn btn-secondary" onClick={leaveRoom}>나가기</button>
            </div>
          </div>

          <div className="game-layout">
            <div className="player-panel">
              <h3>플레이어</h3>
              
              <div className={`player-card cho ${!choPlayer ? 'waiting' : ''}`}>
                <div className="player-nickname">
                  {choPlayer ? `${choPlayer.nickname}${choPlayer.player_id === playerId ? ' (나)' : ''}` : '대기중...'}
                </div>
                <div className="player-side">초 (선공)</div>
                <span className={`player-status ${choPlayer?.is_ready ? 'ready' : 'not-ready'}`}>
                  {choPlayer?.is_ready ? '준비 완료' : '준비 대기'}
                </span>
              </div>

              <div className={`player-card han ${!hanPlayer ? 'waiting' : ''}`}>
                <div className="player-nickname">
                  {hanPlayer ? `${hanPlayer.nickname}${hanPlayer.player_id === playerId ? ' (나)' : ''}` : '대기중...'}
                </div>
                <div className="player-side">한 (후공)</div>
                <span className={`player-status ${hanPlayer?.is_ready ? 'ready' : 'not-ready'}`}>
                  {hanPlayer?.is_ready ? '준비 완료' : '준비 대기'}
                </span>
              </div>

              {!gameInProgress && (
                <>
                  <div className="formation-select">
                    <label>나의 상차림</label>
                    <select value={formation} onChange={(e) => {
                      setFormation(e.target.value);
                      sendMessage({
                        type: 'set_formation',
                        formation: e.target.value,
                      });
                    }}>
                      <option value="마상상마">마상상마 (기본)</option>
                      <option value="상마상마">상마상마</option>
                      <option value="마상마상">마상마상</option>
                      <option value="상마마상">상마마상</option>
                    </select>
                  </div>

                  <button className="btn btn-primary ready-btn" onClick={toggleReady}>
                    {isReady ? '준비 취소' : '준비 완료'}
                  </button>
                </>
              )}

              {gameInProgress && (
                <div className="game-controls">
                  <button className="btn btn-secondary" onClick={offerDraw}>무승부 제안</button>
                  <button className="btn btn-secondary" style={{ background: 'rgba(220, 53, 69, 0.2)', color: '#ff6b7a' }} onClick={resign}>기권</button>
                </div>
              )}
            </div>

            <div className="board-section">
              <div className="board-wrapper">
                <canvas ref={canvasRef} width={CANVAS_WIDTH} height={CANVAS_HEIGHT} onClick={handleCanvasClick} />
              </div>
              <div className="game-status">
                {gameInProgress ? (
                  <>
                    <span className={`turn-indicator ${currentTurn.toLowerCase()}`}></span>
                    {currentTurn === 'CHO' ? '초' : '한'} 차례 {currentTurn === myPlayerSide ? '(내 차례)' : ''}
                  </>
                ) : (
                  '게임이 시작되기를 기다리고 있습니다...'
                )}
              </div>
            </div>

            <div className="chat-panel">
              <div className="chat-header">💬 채팅</div>
              <div className="chat-messages">
                {chatMessages.map((msg, idx) => (
                  <div key={idx} className={`chat-message ${msg.type === 'system' ? 'system' : ''}`}>
                    {msg.type !== 'system' && (
                      <div className={`chat-author ${getSideForPlayer(msg.player_id)}`}>{msg.nickname}</div>
                    )}
                    <div className="chat-text" dangerouslySetInnerHTML={{ __html: escapeHtml(msg.message) }}></div>
                  </div>
                ))}
              </div>
              <div className="chat-input">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyPress={handleChatKeypress}
                  placeholder="메시지 입력..."
                />
                <button className="btn btn-primary chat-send-btn" onClick={sendChat}>전송</button>
              </div>
            </div>
          </div>
        </div>
      )}

      {showCreateModal && (
        <div className="modal-overlay" onClick={() => setShowCreateModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>새 방 만들기</h3>
            <input
              type="text"
              className="modal-input"
              value={createNickname}
              onChange={(e) => setCreateNickname(e.target.value)}
              placeholder="닉네임 입력"
            />
            <label style={{ color: '#b8b8c8', display: 'block', marginBottom: '8px', fontSize: '0.9em' }}>시간 제한 (선택)</label>
            <select className="modal-input" value={timeLimit} onChange={(e) => setTimeLimit(e.target.value)}>
              <option value="">시간 제한 없음</option>
              <option value="300">5분</option>
              <option value="600">10분</option>
              <option value="900">15분</option>
              <option value="1800">30분</option>
            </select>
            <div className="modal-buttons">
              <button className="btn btn-secondary" onClick={() => setShowCreateModal(false)}>취소</button>
              <button className="btn btn-primary" onClick={createRoom}>만들기</button>
            </div>
          </div>
        </div>
      )}

      {showJoinModal && (
        <div className="modal-overlay" onClick={() => setShowJoinModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>방 입장하기</h3>
            <input
              type="text"
              className="modal-input"
              value={joinNickname}
              onChange={(e) => setJoinNickname(e.target.value)}
              placeholder="닉네임 입력"
            />
            <input
              type="text"
              className="modal-input"
              value={joinRoomCode}
              onChange={(e) => setJoinRoomCode(e.target.value.toUpperCase())}
              placeholder="방 코드 입력"
              style={{ textTransform: 'uppercase', letterSpacing: '2px' }}
            />
            <div className="modal-buttons">
              <button className="btn btn-secondary" onClick={() => setShowJoinModal(false)}>취소</button>
              <button className="btn btn-primary" onClick={joinRoomByCode}>입장</button>
            </div>
          </div>
        </div>
      )}

      {showDrawModal && (
        <div className="modal-overlay" onClick={() => setShowDrawModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>무승부 제안</h3>
            <p style={{ textAlign: 'center', color: '#b8b8c8', marginBottom: '20px' }}>
              상대방이 무승부를 제안했습니다.
            </p>
            <div className="modal-buttons">
              <button className="btn btn-secondary" onClick={() => respondDraw(false)}>거절</button>
              <button className="btn btn-primary" onClick={() => respondDraw(true)}>수락</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MultiplayerPage;

