"""Local policy wrapper that bypasses WebSocket and directly calls policy.infer()."""

import logging
from typing import Dict

from typing_extensions import override

from openpi_client import base_policy as _base_policy


class LocalPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by directly calling a local policy object.

    This avoids WebSocket communication entirely and is useful for headless evaluation
    where we don't want network overhead or timeout issues.
    """

    def __init__(self, policy: _base_policy.BasePolicy) -> None:
        """Initialize with a local policy object.

        Args:
            policy: A policy object that implements the BasePolicy interface
                    (typically from openpi.policies.policy.Policy)
        """
        self._policy = policy
        logging.info("LocalPolicy initialized - bypassing WebSocket")

    @override
    def infer(self, obs: Dict) -> Dict:
        """Run inference directly on the local policy."""
        return self._policy.infer(obs)

    @override
    def reset(self) -> None:
        """Reset the policy to its initial state."""
        self._policy.reset()