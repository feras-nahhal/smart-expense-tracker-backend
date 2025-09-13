# app/websocket_manager.py
from typing import Dict, Optional
from fastapi import WebSocket
import asyncio

class ConnectionManager:
    """
    Manages WebSocket connections per user_id.
    Only one active socket is kept per user_id (you can extend to multiple sockets per user later).
    """

    def __init__(self):
        # user_id -> WebSocket
        self.active: Dict[int, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def connect(self, user_id: int, websocket: WebSocket):
        """Register a user's websocket connection. If an old connection exists, close it."""
        async with self._lock:
            old = self.active.get(user_id)
            if old:
                try:
                    await old.close()
                except Exception:
                    pass
            self.active[user_id] = websocket

    async def disconnect(self, user_id: int):
        """Remove a user's connection (if exists)."""
        async with self._lock:
            self.active.pop(user_id, None)

    async def send_personal_message(self, user_id: int, message: str):
        """Send a message to a specific user. If not connected, do nothing."""
        ws = self.active.get(user_id)
        if not ws:
            return
        try:
            await ws.send_text(message)
        except Exception:
            # On failure, remove connection
            async with self._lock:
                self.active.pop(user_id, None)

    async def broadcast_all(self, message: str):
        """Broadcast to all connected users."""
        # snapshot so we don't hold lock while sending
        conns = list(self.active.items())
        for uid, ws in conns:
            try:
                await ws.send_text(message)
            except Exception:
                async with self._lock:
                    self.active.pop(uid, None)

# single shared manager to import
manager = ConnectionManager()
