"""
SYNAPTEX·触链典 — Multi-Agent Shared Memory Protocol

Enables multiple AI agents to share, sync, and selectively access
a common memory pool with permission controls.

Features:
- Shared memory pool with read/write permissions per agent
- Private memory partitions per agent
- Conflict resolution for concurrent writes
- Memory broadcast for system-wide updates
- Agent-specific views with filtered context
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from datetime import datetime
import threading

from synaptex.types import MemoryUnit, AgentIdentity


@dataclass
class MemoryAccessLog:
    """Log entry for memory access auditing."""
    agent_id: str
    memory_id: str
    access_type: str  # "read", "write", "delete"
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    reason: str = ""


class SharedMemoryPool:
    """
    Multi-agent shared memory with permission-based access control.
    
    Architecture:
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │  Agent A     │  │  Agent B     │  │  Agent C     │
    │  (Planner)   │  │  (Executor)  │  │  (Critic)    │
    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
           │                 │                 │
           └─────────┬───────┴─────────┬───────┘
                     ▼                 ▼
           ┌─────────────────────────────────┐
           │  Shared Memory Pool              │
           │  ┌───────────────────────────┐   │
           │  │ Public Partition           │   │
           │  │ (readable by all agents)   │   │
           │  ├───────────────────────────┤   │
           │  │ Private: Agent A           │   │
           │  │ Private: Agent B           │   │
           │  │ Private: Agent C           │   │
           │  └───────────────────────────┘   │
           └─────────────────────────────────┘
    
    Usage:
        pool = SharedMemoryPool()
        pool.register_agent(AgentIdentity(agent_id="planner", role="planning"))
        pool.register_agent(AgentIdentity(agent_id="executor", role="execution"))
        
        mem = MemoryUnit(content_l3="Project deadline is March 15")
        pool.write("planner", mem, scope="public")
        
        # Executor can read it
        results = pool.read("executor", tags=["deadline"])
    """

    def __init__(self):
        self.agents: Dict[str, AgentIdentity] = {}
        self.public_memories: Dict[str, MemoryUnit] = {}
        self.private_memories: Dict[str, Dict[str, MemoryUnit]] = defaultdict(dict)
        self.access_log: List[MemoryAccessLog] = []
        self._lock = threading.RLock()

    def register_agent(self, agent: AgentIdentity):
        """Register an agent with the shared memory pool."""
        self.agents[agent.agent_id] = agent
        if agent.agent_id not in self.private_memories:
            self.private_memories[agent.agent_id] = {}

    def unregister_agent(self, agent_id: str):
        """Remove an agent. Private memories are archived to public."""
        if agent_id in self.agents:
            # Move private memories to public
            for mid, mem in self.private_memories.get(agent_id, {}).items():
                self.public_memories[mid] = mem
            self.private_memories.pop(agent_id, None)
            del self.agents[agent_id]

    def _check_permission(self, agent_id: str, action: str) -> bool:
        """Check if an agent has permission for an action."""
        if agent_id not in self.agents:
            return False
        return action in self.agents[agent_id].permissions

    def write(
        self,
        agent_id: str,
        memory: MemoryUnit,
        scope: str = "public",
    ) -> bool:
        """
        Write a memory to the pool.
        
        Args:
            agent_id: Writing agent
            memory: Memory to store
            scope: "public" (visible to all) or "private" (agent-only)
            
        Returns:
            True if write succeeded
        """
        if not self._check_permission(agent_id, "write"):
            self.access_log.append(MemoryAccessLog(
                agent_id=agent_id, memory_id=memory.id,
                access_type="write", success=False,
                reason="Permission denied",
            ))
            return False

        with self._lock:
            memory.metadata["written_by"] = agent_id
            memory.metadata["written_at"] = datetime.now().isoformat()
            
            if scope == "private":
                self.private_memories[agent_id][memory.id] = memory
                self.agents[agent_id].private_memories.add(memory.id)
            else:
                self.public_memories[memory.id] = memory

        self.access_log.append(MemoryAccessLog(
            agent_id=agent_id, memory_id=memory.id,
            access_type="write",
        ))
        return True

    def read(
        self,
        agent_id: str,
        memory_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        include_private: bool = True,
    ) -> List[MemoryUnit]:
        """
        Read memories from the pool.
        
        Args:
            agent_id: Reading agent
            memory_id: Specific memory to read (optional)
            tags: Filter by tags (optional)
            include_private: Include the agent's own private memories
            
        Returns:
            List of accessible MemoryUnits
        """
        if not self._check_permission(agent_id, "read"):
            return []

        results = []

        # Public memories
        if memory_id:
            if memory_id in self.public_memories:
                results.append(self.public_memories[memory_id])
        else:
            results.extend(self.public_memories.values())

        # Agent's private memories
        if include_private and agent_id in self.private_memories:
            if memory_id:
                if memory_id in self.private_memories[agent_id]:
                    results.append(self.private_memories[agent_id][memory_id])
            else:
                results.extend(self.private_memories[agent_id].values())

        # Filter by tags
        if tags:
            results = [
                m for m in results
                if any(t in m.tags for t in tags)
            ]

        for m in results:
            self.access_log.append(MemoryAccessLog(
                agent_id=agent_id, memory_id=m.id, access_type="read",
            ))

        return results

    def broadcast(self, agent_id: str, memory: MemoryUnit) -> int:
        """
        Broadcast a memory to all agents' contexts.
        High-priority system-wide update.
        
        Returns:
            Number of agents that received the broadcast
        """
        memory.metadata["broadcast_by"] = agent_id
        memory.metadata["broadcast_at"] = datetime.now().isoformat()
        memory.dopamine_weight = max(memory.dopamine_weight, 0.8)
        
        self.public_memories[memory.id] = memory
        return len(self.agents)

    def delete(self, agent_id: str, memory_id: str) -> bool:
        """Delete a memory. Only the original writer or admin can delete."""
        if not self._check_permission(agent_id, "write"):
            return False

        with self._lock:
            # Check public
            if memory_id in self.public_memories:
                mem = self.public_memories[memory_id]
                if mem.metadata.get("written_by") == agent_id or "admin" in self.agents.get(agent_id, AgentIdentity(agent_id="")).permissions:
                    del self.public_memories[memory_id]
                    return True

            # Check private
            if memory_id in self.private_memories.get(agent_id, {}):
                del self.private_memories[agent_id][memory_id]
                self.agents[agent_id].private_memories.discard(memory_id)
                return True

        return False

    def get_agent_view(self, agent_id: str) -> Dict:
        """
        Get a filtered view of the memory pool for a specific agent.
        Returns stats and accessible memory count.
        """
        public_count = len(self.public_memories)
        private_count = len(self.private_memories.get(agent_id, {}))
        
        return {
            "agent_id": agent_id,
            "role": self.agents.get(agent_id, AgentIdentity(agent_id="")).role,
            "public_memories": public_count,
            "private_memories": private_count,
            "total_accessible": public_count + private_count,
            "permissions": list(self.agents.get(agent_id, AgentIdentity(agent_id="")).permissions),
        }

    def get_pool_stats(self) -> Dict:
        """Return overall pool statistics."""
        return {
            "registered_agents": len(self.agents),
            "public_memories": len(self.public_memories),
            "private_partitions": {
                aid: len(mems) for aid, mems in self.private_memories.items()
            },
            "total_access_events": len(self.access_log),
        }
