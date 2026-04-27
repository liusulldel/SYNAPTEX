"""Permissioned shared memory for multi-agent workflows."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import threading
from typing import Dict, List, Optional

from synaptex.types import AgentIdentity, MemoryUnit


@dataclass
class MemoryAccessLog:
    """Audit log entry for shared-memory access."""

    agent_id: str
    memory_id: str
    access_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    reason: str = ""


class SharedMemoryPool:
    """Public/private memory pool with simple per-agent permissions."""

    def __init__(self):
        self.agents: Dict[str, AgentIdentity] = {}
        self.public_memories: Dict[str, MemoryUnit] = {}
        self.private_memories: Dict[str, Dict[str, MemoryUnit]] = defaultdict(dict)
        self.access_log: List[MemoryAccessLog] = []
        self._lock = threading.RLock()

    def register_agent(self, agent: AgentIdentity) -> None:
        self.agents[agent.agent_id] = agent
        self.private_memories.setdefault(agent.agent_id, {})

    def unregister_agent(self, agent_id: str) -> bool:
        if agent_id not in self.agents:
            return False

        with self._lock:
            for memory_id, memory in self.private_memories.get(agent_id, {}).items():
                self.public_memories[memory_id] = memory
            self.private_memories.pop(agent_id, None)
            del self.agents[agent_id]
        return True

    def _check_permission(self, agent_id: str, action: str) -> bool:
        agent = self.agents.get(agent_id)
        return bool(agent and action in agent.permissions)

    def write(self, agent_id: str, memory: MemoryUnit, scope: str = "public") -> bool:
        """Write memory to public or private scope."""

        if scope not in {"public", "private"}:
            raise ValueError("scope must be 'public' or 'private'")

        if not self._check_permission(agent_id, "write"):
            self.access_log.append(
                MemoryAccessLog(
                    agent_id=agent_id,
                    memory_id=memory.id,
                    access_type="write",
                    success=False,
                    reason="permission_denied",
                )
            )
            return False

        with self._lock:
            memory.metadata["written_by"] = agent_id
            memory.metadata["written_at"] = datetime.now().isoformat()

            if scope == "private":
                self.private_memories[agent_id][memory.id] = memory
                self.agents[agent_id].private_memories.add(memory.id)
            else:
                self.public_memories[memory.id] = memory

        self.access_log.append(MemoryAccessLog(agent_id, memory.id, "write"))
        return True

    def read(
        self,
        agent_id: str,
        memory_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        include_private: bool = True,
    ) -> List[MemoryUnit]:
        """Read accessible memories for an agent."""

        if not self._check_permission(agent_id, "read"):
            self.access_log.append(
                MemoryAccessLog(
                    agent_id=agent_id,
                    memory_id=memory_id or "*",
                    access_type="read",
                    success=False,
                    reason="permission_denied",
                )
            )
            return []

        results: List[MemoryUnit] = []
        if memory_id:
            if memory_id in self.public_memories:
                results.append(self.public_memories[memory_id])
        else:
            results.extend(self.public_memories.values())

        if include_private and agent_id in self.private_memories:
            private = self.private_memories[agent_id]
            if memory_id:
                if memory_id in private:
                    results.append(private[memory_id])
            else:
                results.extend(private.values())

        if tags:
            results = [memory for memory in results if any(tag in memory.tags for tag in tags)]

        for memory in results:
            self.access_log.append(MemoryAccessLog(agent_id, memory.id, "read"))
        return results

    def broadcast(self, agent_id: str, memory: MemoryUnit) -> int:
        """Write a high-priority public memory visible to all registered agents."""

        if not self._check_permission(agent_id, "write"):
            return 0

        memory.metadata["broadcast_by"] = agent_id
        memory.metadata["broadcast_at"] = datetime.now().isoformat()
        memory.dopamine_weight = max(memory.dopamine_weight, 0.8)
        self.public_memories[memory.id] = memory
        self.access_log.append(MemoryAccessLog(agent_id, memory.id, "broadcast"))
        return len(self.agents)

    def delete(self, agent_id: str, memory_id: str) -> bool:
        """Delete memory if the agent wrote it or has admin permission."""

        if not self._check_permission(agent_id, "write"):
            return False

        agent = self.agents.get(agent_id)
        is_admin = bool(agent and "admin" in agent.permissions)

        with self._lock:
            memory = self.public_memories.get(memory_id)
            if memory and (memory.metadata.get("written_by") == agent_id or is_admin):
                del self.public_memories[memory_id]
                self.access_log.append(MemoryAccessLog(agent_id, memory_id, "delete"))
                return True

            if memory_id in self.private_memories.get(agent_id, {}):
                del self.private_memories[agent_id][memory_id]
                self.agents[agent_id].private_memories.discard(memory_id)
                self.access_log.append(MemoryAccessLog(agent_id, memory_id, "delete"))
                return True

        return False

    def get_agent_view(self, agent_id: str) -> Dict:
        agent = self.agents.get(agent_id, AgentIdentity(agent_id=agent_id))
        public_count = len(self.public_memories)
        private_count = len(self.private_memories.get(agent_id, {}))
        return {
            "agent_id": agent_id,
            "role": agent.role,
            "public_memories": public_count,
            "private_memories": private_count,
            "total_accessible": public_count + private_count,
            "permissions": sorted(agent.permissions),
        }

    def get_pool_stats(self) -> Dict:
        return {
            "registered_agents": len(self.agents),
            "public_memories": len(self.public_memories),
            "private_partitions": {
                agent_id: len(memories) for agent_id, memories in self.private_memories.items()
            },
            "total_access_events": len(self.access_log),
        }
