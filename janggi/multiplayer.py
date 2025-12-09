"""WebSocket-based multiplayer support for Janggi.

Provides real-time game rooms for online play between two players.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List, Any, Set
from time import time

from fastapi import WebSocket, WebSocketDisconnect


class RoomStatus(Enum):
    """Room status states."""
    WAITING = "waiting"        # Waiting for second player
    PLAYING = "playing"        # Game in progress
    FINISHED = "finished"      # Game ended
    ABANDONED = "abandoned"    # Player disconnected


class PlayerSide(Enum):
    """Player side in the game."""
    CHO = "CHO"  # First to move
    HAN = "HAN"  # Second to move


@dataclass
class Player:
    """Represents a connected player."""
    player_id: str
    nickname: str
    websocket: WebSocket
    side: Optional[PlayerSide] = None
    connected_at: float = field(default_factory=time)
    last_ping: float = field(default_factory=time)
    is_ready: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without websocket)."""
        return {
            "player_id": self.player_id,
            "nickname": self.nickname,
            "side": self.side.value if self.side else None,
            "is_ready": self.is_ready,
        }


@dataclass
class GameRoom:
    """Represents a multiplayer game room."""
    room_id: str
    created_at: float = field(default_factory=time)
    status: RoomStatus = RoomStatus.WAITING
    players: Dict[str, Player] = field(default_factory=dict)  # player_id -> Player
    host_id: Optional[str] = None  # First player who created the room
    
    # Game state
    board_state: Optional[Dict] = None  # Serialized board state
    current_turn: PlayerSide = PlayerSide.CHO
    move_history: List[Dict] = field(default_factory=list)
    
    # Formation choices
    cho_formation: str = "마상상마"
    han_formation: str = "마상상마"
    
    # Timing
    time_limit: Optional[int] = None  # seconds per player (None = no limit)
    cho_time_remaining: Optional[float] = None
    han_time_remaining: Optional[float] = None
    last_move_time: Optional[float] = None
    
    def get_player_by_side(self, side: PlayerSide) -> Optional[Player]:
        """Get player by their side."""
        for player in self.players.values():
            if player.side == side:
                return player
        return None
    
    def get_opponent(self, player_id: str) -> Optional[Player]:
        """Get the opponent of a player."""
        player = self.players.get(player_id)
        if not player or not player.side:
            return None
        
        opponent_side = PlayerSide.HAN if player.side == PlayerSide.CHO else PlayerSide.CHO
        return self.get_player_by_side(opponent_side)
    
    def is_full(self) -> bool:
        """Check if room has two players."""
        return len(self.players) >= 2
    
    def is_empty(self) -> bool:
        """Check if room has no players."""
        return len(self.players) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "room_id": self.room_id,
            "status": self.status.value,
            "players": {pid: p.to_dict() for pid, p in self.players.items()},
            "host_id": self.host_id,
            "current_turn": self.current_turn.value,
            "cho_formation": self.cho_formation,
            "han_formation": self.han_formation,
            "time_limit": self.time_limit,
            "cho_time_remaining": self.cho_time_remaining,
            "han_time_remaining": self.han_time_remaining,
            "move_count": len(self.move_history),
        }


class RoomManager:
    """Manages multiplayer game rooms."""
    
    def __init__(self):
        self.rooms: Dict[str, GameRoom] = {}
        self.player_to_room: Dict[str, str] = {}  # player_id -> room_id
        self._lock = asyncio.Lock()
    
    async def create_room(self, host: Player, time_limit: Optional[int] = None) -> GameRoom:
        """Create a new game room."""
        async with self._lock:
            room_id = str(uuid.uuid4())[:8].upper()
            
            # Ensure unique room ID
            while room_id in self.rooms:
                room_id = str(uuid.uuid4())[:8].upper()
            
            room = GameRoom(
                room_id=room_id,
                host_id=host.player_id,
                time_limit=time_limit,
            )
            
            if time_limit:
                room.cho_time_remaining = float(time_limit)
                room.han_time_remaining = float(time_limit)
            
            # Host joins as CHO by default
            host.side = PlayerSide.CHO
            room.players[host.player_id] = host
            
            self.rooms[room_id] = room
            self.player_to_room[host.player_id] = room_id
            
            return room
    
    async def join_room(self, room_id: str, player: Player) -> Optional[GameRoom]:
        """Join an existing room. Returns room if successful, None otherwise."""
        async with self._lock:
            room = self.rooms.get(room_id)
            
            if not room:
                return None
            
            if room.is_full():
                return None
            
            if room.status != RoomStatus.WAITING:
                return None
            
            # Assign HAN side to joining player
            player.side = PlayerSide.HAN
            room.players[player.player_id] = player
            self.player_to_room[player.player_id] = room_id
            
            return room
    
    async def leave_room(self, player_id: str) -> Optional[GameRoom]:
        """Remove a player from their room. Returns the room."""
        async with self._lock:
            room_id = self.player_to_room.pop(player_id, None)
            if not room_id:
                return None
            
            room = self.rooms.get(room_id)
            if not room:
                return None
            
            player = room.players.pop(player_id, None)
            
            if room.is_empty():
                # Delete empty room
                del self.rooms[room_id]
                return None
            
            # If game was in progress, mark as abandoned
            if room.status == RoomStatus.PLAYING:
                room.status = RoomStatus.ABANDONED
            
            return room
    
    def get_room(self, room_id: str) -> Optional[GameRoom]:
        """Get a room by ID."""
        return self.rooms.get(room_id)
    
    def get_player_room(self, player_id: str) -> Optional[GameRoom]:
        """Get the room a player is in."""
        room_id = self.player_to_room.get(player_id)
        if room_id:
            return self.rooms.get(room_id)
        return None
    
    def get_available_rooms(self) -> List[Dict[str, Any]]:
        """Get list of rooms waiting for players."""
        return [
            room.to_dict() 
            for room in self.rooms.values() 
            if room.status == RoomStatus.WAITING and not room.is_full()
        ]
    
    async def cleanup_stale_rooms(self, max_idle_seconds: int = 3600):
        """Remove rooms that have been idle for too long."""
        async with self._lock:
            current_time = time()
            to_remove = []
            
            for room_id, room in self.rooms.items():
                # Check if all players have disconnected
                all_disconnected = all(
                    current_time - p.last_ping > 60 
                    for p in room.players.values()
                )
                
                # Or room is too old
                is_stale = current_time - room.created_at > max_idle_seconds
                
                if all_disconnected or (is_stale and room.status != RoomStatus.PLAYING):
                    to_remove.append(room_id)
            
            for room_id in to_remove:
                room = self.rooms.pop(room_id, None)
                if room:
                    for player_id in room.players:
                        self.player_to_room.pop(player_id, None)


class ConnectionManager:
    """Manages WebSocket connections for multiplayer."""
    
    def __init__(self):
        self.room_manager = RoomManager()
        self.active_connections: Dict[str, WebSocket] = {}  # player_id -> websocket
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, player_id: str) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections[player_id] = websocket
    
    async def disconnect(self, player_id: str) -> None:
        """Handle player disconnection."""
        async with self._lock:
            self.active_connections.pop(player_id, None)
        
        # Leave any room the player was in
        room = await self.room_manager.leave_room(player_id)
        
        if room:
            # Notify remaining player
            await self.broadcast_to_room(room.room_id, {
                "type": "player_left",
                "player_id": player_id,
                "room": room.to_dict(),
            })
    
    async def send_personal(self, player_id: str, message: Dict) -> bool:
        """Send a message to a specific player."""
        websocket = self.active_connections.get(player_id)
        if websocket:
            try:
                await websocket.send_json(message)
                return True
            except Exception:
                pass
        return False
    
    async def broadcast_to_room(self, room_id: str, message: Dict, exclude: Optional[str] = None) -> None:
        """Broadcast a message to all players in a room."""
        room = self.room_manager.get_room(room_id)
        if not room:
            return
        
        for player_id in room.players:
            if player_id != exclude:
                await self.send_personal(player_id, message)
    
    async def handle_message(self, player_id: str, data: Dict) -> None:
        """Handle incoming WebSocket message."""
        msg_type = data.get("type")
        
        handlers = {
            "create_room": self._handle_create_room,
            "join_room": self._handle_join_room,
            "leave_room": self._handle_leave_room,
            "set_formation": self._handle_set_formation,
            "set_ready": self._handle_set_ready,
            "make_move": self._handle_make_move,
            "chat": self._handle_chat,
            "ping": self._handle_ping,
            "get_rooms": self._handle_get_rooms,
            "offer_draw": self._handle_offer_draw,
            "respond_draw": self._handle_respond_draw,
            "resign": self._handle_resign,
        }
        
        handler = handlers.get(msg_type)
        if handler:
            await handler(player_id, data)
        else:
            await self.send_personal(player_id, {
                "type": "error",
                "message": f"Unknown message type: {msg_type}",
            })
    
    async def _handle_create_room(self, player_id: str, data: Dict) -> None:
        """Handle room creation request."""
        nickname = data.get("nickname", f"Player_{player_id[:6]}")
        time_limit = data.get("time_limit")  # seconds, or None
        
        websocket = self.active_connections.get(player_id)
        if not websocket:
            return
        
        player = Player(
            player_id=player_id,
            nickname=nickname,
            websocket=websocket,
        )
        
        room = await self.room_manager.create_room(player, time_limit)
        
        await self.send_personal(player_id, {
            "type": "room_created",
            "room": room.to_dict(),
            "your_side": player.side.value,
        })
    
    async def _handle_join_room(self, player_id: str, data: Dict) -> None:
        """Handle room join request."""
        room_id = data.get("room_id")
        nickname = data.get("nickname", f"Player_{player_id[:6]}")
        
        websocket = self.active_connections.get(player_id)
        if not websocket:
            return
        
        player = Player(
            player_id=player_id,
            nickname=nickname,
            websocket=websocket,
        )
        
        room = await self.room_manager.join_room(room_id, player)
        
        if room:
            # Notify joining player
            await self.send_personal(player_id, {
                "type": "room_joined",
                "room": room.to_dict(),
                "your_side": player.side.value,
            })
            
            # Notify existing player
            await self.broadcast_to_room(room_id, {
                "type": "player_joined",
                "player": player.to_dict(),
                "room": room.to_dict(),
            }, exclude=player_id)
        else:
            await self.send_personal(player_id, {
                "type": "error",
                "message": "Could not join room. It may be full or doesn't exist.",
            })
    
    async def _handle_leave_room(self, player_id: str, data: Dict) -> None:
        """Handle room leave request."""
        room = await self.room_manager.leave_room(player_id)
        
        await self.send_personal(player_id, {
            "type": "room_left",
        })
        
        if room:
            await self.broadcast_to_room(room.room_id, {
                "type": "player_left",
                "player_id": player_id,
                "room": room.to_dict(),
            })
    
    async def _handle_set_formation(self, player_id: str, data: Dict) -> None:
        """Handle formation selection."""
        formation = data.get("formation", "마상상마")
        room = self.room_manager.get_player_room(player_id)
        
        if not room:
            return
        
        player = room.players.get(player_id)
        if not player or not player.side:
            return
        
        if player.side == PlayerSide.CHO:
            room.cho_formation = formation
        else:
            room.han_formation = formation
        
        await self.broadcast_to_room(room.room_id, {
            "type": "formation_changed",
            "player_id": player_id,
            "side": player.side.value,
            "formation": formation,
            "room": room.to_dict(),
        })
    
    async def _handle_set_ready(self, player_id: str, data: Dict) -> None:
        """Handle player ready status."""
        is_ready = data.get("is_ready", True)
        room = self.room_manager.get_player_room(player_id)
        
        if not room:
            return
        
        player = room.players.get(player_id)
        if not player:
            return
        
        player.is_ready = is_ready
        
        await self.broadcast_to_room(room.room_id, {
            "type": "ready_changed",
            "player_id": player_id,
            "is_ready": is_ready,
            "room": room.to_dict(),
        })
        
        # Check if both players are ready
        if room.is_full() and all(p.is_ready for p in room.players.values()):
            await self._start_game(room)
    
    async def _start_game(self, room: GameRoom) -> None:
        """Start the game when both players are ready."""
        room.status = RoomStatus.PLAYING
        room.last_move_time = time()
        
        await self.broadcast_to_room(room.room_id, {
            "type": "game_started",
            "room": room.to_dict(),
            "cho_formation": room.cho_formation,
            "han_formation": room.han_formation,
        })
    
    async def _handle_make_move(self, player_id: str, data: Dict) -> None:
        """Handle player move."""
        room = self.room_manager.get_player_room(player_id)
        
        if not room or room.status != RoomStatus.PLAYING:
            await self.send_personal(player_id, {
                "type": "error",
                "message": "Game is not in progress.",
            })
            return
        
        player = room.players.get(player_id)
        if not player or player.side != room.current_turn:
            await self.send_personal(player_id, {
                "type": "error",
                "message": "Not your turn.",
            })
            return
        
        from_square = data.get("from_square")
        to_square = data.get("to_square")
        
        if not from_square or not to_square:
            await self.send_personal(player_id, {
                "type": "error",
                "message": "Invalid move data.",
            })
            return
        
        # Update timing
        current_time = time()
        if room.time_limit and room.last_move_time:
            elapsed = current_time - room.last_move_time
            if player.side == PlayerSide.CHO:
                room.cho_time_remaining = max(0, (room.cho_time_remaining or 0) - elapsed)
                if room.cho_time_remaining <= 0:
                    await self._handle_timeout(room, PlayerSide.CHO)
                    return
            else:
                room.han_time_remaining = max(0, (room.han_time_remaining or 0) - elapsed)
                if room.han_time_remaining <= 0:
                    await self._handle_timeout(room, PlayerSide.HAN)
                    return
        
        room.last_move_time = current_time
        
        # Record move
        move_record = {
            "move_number": len(room.move_history) + 1,
            "side": player.side.value,
            "from": from_square,
            "to": to_square,
            "timestamp": current_time,
        }
        room.move_history.append(move_record)
        
        # Switch turn
        room.current_turn = PlayerSide.HAN if room.current_turn == PlayerSide.CHO else PlayerSide.CHO
        
        # Broadcast move to all players
        await self.broadcast_to_room(room.room_id, {
            "type": "move_made",
            "move": move_record,
            "room": room.to_dict(),
        })
    
    async def _handle_timeout(self, room: GameRoom, timed_out_side: PlayerSide) -> None:
        """Handle player timeout."""
        winner_side = PlayerSide.HAN if timed_out_side == PlayerSide.CHO else PlayerSide.CHO
        room.status = RoomStatus.FINISHED
        
        await self.broadcast_to_room(room.room_id, {
            "type": "game_over",
            "reason": "timeout",
            "winner": winner_side.value,
            "loser": timed_out_side.value,
            "room": room.to_dict(),
        })
    
    async def _handle_chat(self, player_id: str, data: Dict) -> None:
        """Handle chat message."""
        message = data.get("message", "")
        room = self.room_manager.get_player_room(player_id)
        
        if not room or not message:
            return
        
        player = room.players.get(player_id)
        if not player:
            return
        
        await self.broadcast_to_room(room.room_id, {
            "type": "chat",
            "player_id": player_id,
            "nickname": player.nickname,
            "message": message[:500],  # Limit message length
            "timestamp": time(),
        })
    
    async def _handle_ping(self, player_id: str, data: Dict) -> None:
        """Handle ping for connection keep-alive."""
        room = self.room_manager.get_player_room(player_id)
        
        if room:
            player = room.players.get(player_id)
            if player:
                player.last_ping = time()
        
        await self.send_personal(player_id, {
            "type": "pong",
            "timestamp": time(),
        })
    
    async def _handle_get_rooms(self, player_id: str, data: Dict) -> None:
        """Handle request for available rooms."""
        rooms = self.room_manager.get_available_rooms()
        
        await self.send_personal(player_id, {
            "type": "room_list",
            "rooms": rooms,
        })
    
    async def _handle_offer_draw(self, player_id: str, data: Dict) -> None:
        """Handle draw offer."""
        room = self.room_manager.get_player_room(player_id)
        
        if not room or room.status != RoomStatus.PLAYING:
            return
        
        opponent = room.get_opponent(player_id)
        if opponent:
            await self.send_personal(opponent.player_id, {
                "type": "draw_offered",
                "from_player_id": player_id,
            })
    
    async def _handle_respond_draw(self, player_id: str, data: Dict) -> None:
        """Handle response to draw offer."""
        accept = data.get("accept", False)
        room = self.room_manager.get_player_room(player_id)
        
        if not room or room.status != RoomStatus.PLAYING:
            return
        
        if accept:
            room.status = RoomStatus.FINISHED
            await self.broadcast_to_room(room.room_id, {
                "type": "game_over",
                "reason": "draw_agreed",
                "winner": None,
                "room": room.to_dict(),
            })
        else:
            opponent = room.get_opponent(player_id)
            if opponent:
                await self.send_personal(opponent.player_id, {
                    "type": "draw_declined",
                })
    
    async def _handle_resign(self, player_id: str, data: Dict) -> None:
        """Handle player resignation."""
        room = self.room_manager.get_player_room(player_id)
        
        if not room or room.status != RoomStatus.PLAYING:
            return
        
        player = room.players.get(player_id)
        if not player or not player.side:
            return
        
        room.status = RoomStatus.FINISHED
        winner_side = PlayerSide.HAN if player.side == PlayerSide.CHO else PlayerSide.CHO
        
        await self.broadcast_to_room(room.room_id, {
            "type": "game_over",
            "reason": "resignation",
            "winner": winner_side.value,
            "loser": player.side.value,
            "room": room.to_dict(),
        })


# Global connection manager instance
_connection_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """Get or create the global ConnectionManager instance."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager

