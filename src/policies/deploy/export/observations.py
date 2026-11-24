"""The observations an artifact must know: those its bookkeeping reads."""

from policies.deploy.export.source import Exportable
from policies.deploy.inference.sources import MissingObservationsError, tracker_sources


def observations(source: Exportable) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    """Shapes and dtypes of the observations trackers read. Others are left out.

    Raises `ValueError` when a tracker reads an observation of unknown shape.
    """
    actor = source.actor
    known, dtypes = source.observation_shapes, source.observation_dtypes
    try:
        names = tracker_sources(actor.spec.bookkeeping, actor.decoder, known, dtypes)
    except MissingObservationsError as error:
        raise ValueError(
            f"cannot export {actor.spec.name!r}: {error}. Act once with the learner, or export "
            "`ExportSource(actor, observation_shapes, observation_dtypes)`."
        ) from None
    return (
        {name: tuple(known[name]) for name in names},
        {name: dtypes[name] for name in names if name in dtypes},
    )
