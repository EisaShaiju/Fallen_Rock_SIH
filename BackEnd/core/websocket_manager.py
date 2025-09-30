import json
from typing import List, Dict
from fastapi import WebSocket

class ConnectionManager:
    """
    Manages active WebSocket connections in a multi-tenant environment.
    Connections are stored in a dictionary keyed by mine_id.
    """
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, mine_id: str):
        """Accepts a new WebSocket connection and adds it to the list for the specified mine."""
        await websocket.accept()
        if mine_id not in self.active_connections:
            self.active_connections[mine_id] = []
        self.active_connections[mine_id].append(websocket)

    def disconnect(self, websocket: WebSocket, mine_id: str):
        """Removes a WebSocket connection from the list."""
        if mine_id in self.active_connections:
            self.active_connections[mine_id].remove(websocket)
            if not self.active_connections[mine_id]:
                del self.active_connections[mine_id]

    async def broadcast_to_mine(self, mine_id: str, data: dict):
        """Broadcasts a JSON message to all clients connected to a specific mine."""
        message = json.dumps(data)
        if mine_id in self.active_connections:
            for connection in self.active_connections[mine_id]:
                await connection.send_text(message)

# Create a singleton instance to be used across the application
manager = ConnectionManager()
