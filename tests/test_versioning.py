"""Tests for agent versioning."""

from agentdeploy.core.versioning import VersionManager


class TestVersionManager:
    def setup_method(self):
        self.mgr = VersionManager("test-agent")

    def test_deploy_first_version(self):
        v = self.mgr.deploy("v1", {"model": "haiku"})
        assert v.status == "active"
        assert v.traffic_percent == 100
        assert self.mgr.get_active().version_id == "v1"

    def test_deploy_replaces_active(self):
        self.mgr.deploy("v1", {"model": "haiku"})
        self.mgr.deploy("v2", {"model": "sonnet"})
        active = self.mgr.get_active()
        assert active.version_id == "v2"
        # v1 should be retired
        versions = self.mgr.list_versions()
        v1 = [v for v in versions if v.version_id == "v1"][0]
        assert v1.status == "retired"

    def test_deploy_canary(self):
        self.mgr.deploy("v1", {"model": "haiku"})
        self.mgr.deploy_canary("v2", {"model": "sonnet"}, traffic_percent=20)
        canary = self.mgr.get_canary()
        assert canary.version_id == "v2"
        assert canary.traffic_percent == 20
        active = self.mgr.get_active()
        assert active.traffic_percent == 80

    def test_promote_canary(self):
        self.mgr.deploy("v1", {})
        self.mgr.deploy_canary("v2", {}, traffic_percent=10)
        assert self.mgr.promote_canary()
        assert self.mgr.get_active().version_id == "v2"
        assert self.mgr.get_canary() is None

    def test_rollback_canary(self):
        self.mgr.deploy("v1", {})
        self.mgr.deploy_canary("v2", {}, traffic_percent=10)
        rolled = self.mgr.rollback()
        assert rolled == "v1"
        assert self.mgr.get_canary() is None
        assert self.mgr.get_active().traffic_percent == 100

    def test_rollback_active(self):
        self.mgr.deploy("v1", {})
        self.mgr.deploy("v2", {})
        rolled = self.mgr.rollback()
        assert rolled == "v1"
        assert self.mgr.get_active().version_id == "v1"

    def test_rollback_no_previous(self):
        self.mgr.deploy("v1", {})
        assert self.mgr.rollback() is None

    def test_route_request_no_canary(self):
        self.mgr.deploy("v1", {})
        assert self.mgr.route_request() == "v1"

    def test_route_request_with_canary(self):
        self.mgr.deploy("v1", {})
        self.mgr.deploy_canary("v2", {}, traffic_percent=50)
        # With 50% traffic, should route to both
        results = set()
        for _ in range(100):
            results.add(self.mgr.route_request())
        assert "v1" in results
        assert "v2" in results

    def test_version_count(self):
        self.mgr.deploy("v1", {})
        self.mgr.deploy("v2", {})
        self.mgr.deploy_canary("v3", {})
        assert self.mgr.version_count == 3
