"""The spec as an artifact embeds it."""

from policies.spec import PolicySpec


def canonical_spec(spec: PolicySpec) -> PolicySpec:
    """Revalidated from its JSON form, so its fingerprint is that of what gets embedded.

    A spec changed with `model_copy(update=...)` skips validation and may not be canonical.
    """
    return PolicySpec.model_validate(spec.model_dump(mode="json", by_alias=True, round_trip=True))
