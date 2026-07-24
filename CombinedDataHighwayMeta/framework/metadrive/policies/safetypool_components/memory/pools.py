"""Public SafetyPool memory assembled from single-responsibility mixins."""

from __future__ import annotations

from .candidates import PoolCandidateManagementMixin
from .capacity import PoolCapacityMixin
from .diagnostics import PoolDiagnosticsMixin
from .matching import PoolMatchingMixin
from .outcomes import PoolOutcomeMixin
from .selection import PoolActionSelectionMixin
from .state_processing import PoolStateProcessingMixin
from .storage import PoolStorageMixin


class SimilarStateActionPools(
    PoolDiagnosticsMixin,
    PoolActionSelectionMixin,
    PoolStateProcessingMixin,
    PoolCandidateManagementMixin,
    PoolOutcomeMixin,
    PoolCapacityMixin,
    PoolMatchingMixin,
    PoolStorageMixin,
):
    """Lifecycle memory with bounded retained hazard evidence.

    Each mixin owns one concern. The public class preserves the
    original API and state layout, so training code and saved
    diagnostics remain compatible with the monolithic policy.
    """

    CANDIDATE = "CANDIDATE"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"
    HAZARD = "HAZARD"
    POST_FIRST_PASS_GREEDY_PROBABILITY = 0.80
