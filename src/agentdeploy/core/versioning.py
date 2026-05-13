"""Agent versioning - immutable versions with rollback."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentVersion:
    """An immutable agent version."""

    version_id: str
    agent_name: str
    config_snapshot: Dict[str, Any]
    deployed_at: float = field(default_factory=time.time)
    deployed_by: str = "system"
    status: str = "active"  # active, canary, retired, rolled_back
    traffic_percent: int = 100
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "agent_name": self.agent_name,
            "status": self.status,
            "traffic_percent": self.traffic_percent,
            "deployed_at": self.deployed_at,
        }


class VersionManager:
    """Manages agent versions with canary and rollback.

    Usage:
        mgr = VersionManager("my-agent")
        mgr.deploy("v1", config)
        mgr.deploy_canary("v2", config, traffic_percent=10)
        mgr.promote_canary()  # v2 becomes active
        mgr.rollback()  # back to v1
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self._versions: List[AgentVersion] = []
        self._active_version: Optional[str] = None
        self._canary_version: Optional[str] = None

    def deploy(self, version_id: str, config: Dict[str, Any], deployed_by: str = "system") -> AgentVersion:
        """Deploy a new version as active."""
        # Retire current active
        if self._active_version:
            for v in self._versions:
                if v.version_id == self._active_version:
                    v.status = "retired"
                    v.traffic_percent = 0

        version = AgentVersion(
            version_id=version_id, agent_name=self.agent_name,
            config_snapshot=config, deployed_by=deployed_by,
            status="active", traffic_percent=100,
        )
        self._versions.append(version)
        self._active_version = version_id
        self._canary_version = None
        return version

    def deploy_canary(self, version_id: str, config: Dict[str, Any], traffic_percent: int = 10) -> AgentVersion:
        """Deploy a canary version with partial traffic."""
        version = AgentVersion(
            version_id=version_id, agent_name=self.agent_name,
            config_snapshot=config, status="canary",
            traffic_percent=traffic_percent,
        )
        self._versions.append(version)
        self._canary_version = version_id

        # Reduce active traffic
        if self._active_version:
            for v in self._versions:
                if v.version_id == self._active_version:
                    v.traffic_percent = 100 - traffic_percent

        return version

    def promote_canary(self) -> bool:
        """Promote canary to active."""
        if not self._canary_version:
            return False

        # Retire old active
        if self._active_version:
            for v in self._versions:
                if v.version_id == self._active_version:
                    v.status = "retired"
                    v.traffic_percent = 0

        # Promote canary
        for v in self._versions:
            if v.version_id == self._canary_version:
                v.status = "active"
                v.traffic_percent = 100

        self._active_version = self._canary_version
        self._canary_version = None
        return True

    def rollback(self) -> Optional[str]:
        """Rollback to previous version."""
        # Cancel canary if active
        if self._canary_version:
            for v in self._versions:
                if v.version_id == self._canary_version:
                    v.status = "rolled_back"
                    v.traffic_percent = 0
            self._canary_version = None
            # Restore active to 100%
            if self._active_version:
                for v in self._versions:
                    if v.version_id == self._active_version:
                        v.traffic_percent = 100
            return self._active_version

        # Rollback active to previous
        retired = [v for v in self._versions if v.status == "retired"]
        if not retired:
            return None

        previous = retired[-1]

        # Retire current
        if self._active_version:
            for v in self._versions:
                if v.version_id == self._active_version:
                    v.status = "rolled_back"
                    v.traffic_percent = 0

        # Restore previous
        previous.status = "active"
        previous.traffic_percent = 100
        self._active_version = previous.version_id
        return previous.version_id

    def get_active(self) -> Optional[AgentVersion]:
        """Get currently active version."""
        for v in self._versions:
            if v.version_id == self._active_version:
                return v
        return None

    def get_canary(self) -> Optional[AgentVersion]:
        """Get canary version if active."""
        for v in self._versions:
            if v.version_id == self._canary_version:
                return v
        return None

    def route_request(self) -> str:
        """Route a request to the appropriate version (based on traffic split)."""
        import random
        if self._canary_version:
            canary = self.get_canary()
            if canary and random.randint(1, 100) <= canary.traffic_percent:
                return self._canary_version
        return self._active_version or ""

    def list_versions(self) -> List[AgentVersion]:
        """List all versions."""
        return list(self._versions)

    @property
    def version_count(self) -> int:
        return len(self._versions)
