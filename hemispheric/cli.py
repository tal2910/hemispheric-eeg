"""Orchestrator entrypoint: parse filters, plan chunks, run provider + consumer.

------------------------------------------------------------------------------
Configuration from config.yaml (overridable per-run via CLI flags):
------------------------------------------------------------------------------
  --data-dir DIR           ←  dataset.data_dir
  --ports PORTS [PORTS...] ←  runtime.ports
  --host HOST              ←  runtime.host
  --no-consumer            ←  runtime.consumer_mode == 'none'
  --consumer-cmd CMD       ←  runtime.consumer_cmd
  --seed INT               ←  runtime.seed
  --log-level LEVEL        ←  runtime.log_level

Filter (config.yaml's `filter:` section) — CLI flags override per field:
  --gender         ←  filter.gender (OR)
  --age            ←  filter.age (AND)
  --name           ←  filter.names (OR)
  --visit-id       ←  filter.visit_ids (OR)
  --person-id      ←  filter.person_ids (OR)
  --dominant-hand  ←  filter.dominant_hand (OR)
  --wears-glasses  ←  filter.wears_glasses
  --no-glasses     ←  filter.wears_glasses
------------------------------------------------------------------------------

Typical use:

    # Use the filter defined in config.yaml
    python run_training.py

    # Or override individual filter fields for a one-off run
    python run_training.py --gender male --age "<40"

    # Or swap the entire config (different experiment, same code)
    HEMISPHERIC_CONFIG=experiments/cohort-A.yaml python run_training.py

By default the orchestrator spawns the data team's consumer.py as a
subprocess; pass --no-consumer to run the provider alone (e.g. when the
consumer runs on another host).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
import shlex
import signal
import sys
from pathlib import Path

from .chunking import plan_chunks, shard_round_robin
from .config import load_config
from .filters import build_filter
from .metadata import load_all_visits
from .preflight import build_visit_db, check_data_dir
from .provider import run_provider


def _build_arg_parser() -> argparse.ArgumentParser:
    cfg = load_config()
    p = argparse.ArgumentParser(
        prog="run_training.py",
        description="Orchestrate a deep-learning training run over filtered EEG data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=cfg.dataset.data_dir,
        help=f"Directory containing matching .npy + .json file pairs. "
             f"Default from config.yaml: {cfg.dataset.data_dir}",
    )
    p.add_argument(
        "--ports",
        type=int,
        nargs="+",
        default=list(cfg.runtime.ports),
        help=f"One TCP port per consumer worker. Chunks are sharded round-robin. "
             f"Default from config.yaml: {' '.join(str(p) for p in cfg.runtime.ports)}",
    )
    p.add_argument(
        "--host",
        default=cfg.runtime.host,
        help=f"Host/interface for the provider to bind. "
             f"Default from config.yaml: {cfg.runtime.host}",
    )

    # Filter flags. Defaults come from config.yaml's `filter:` section.
    # Pass any of these to override the config for this run.
    # (Categorical flags are repeatable for OR; --age is repeatable for AND.)
    p.add_argument("--gender", action="append", default=None,
                   help=f"Gender to include. Repeat for OR. "
                        f"Default from config.yaml: {list(cfg.filter.gender) or 'any'}")
    p.add_argument("--age", action="append", default=None,
                   help=f"Age constraint. Repeat for AND. e.g. --age \">20\" --age \"<=40\". "
                        f"Default from config.yaml: {list(cfg.filter.age) or 'any'}")
    p.add_argument("--name", action="append", default=None,
                   help="Person name to include (exact match). Repeat for OR.")
    p.add_argument("--visit-id", action="append", default=None,
                   help="Specific visit UUID to include. Repeat for OR.")
    p.add_argument("--person-id", action="append", default=None,
                   help="Specific person UUID to include. Repeat for OR.")
    p.add_argument("--dominant-hand", action="append", default=None,
                   choices=["right", "left", "ambidextrous"],
                   help=f"Dominant hand to include. Repeat for OR. "
                        f"Default from config.yaml: {list(cfg.filter.dominant_hand) or 'any'}")

    glasses = p.add_mutually_exclusive_group()
    glasses.add_argument("--wears-glasses", dest="wears_glasses",
                         action="store_const", const=True,
                         help="Only include subjects who wear glasses.")
    glasses.add_argument("--no-glasses", dest="wears_glasses",
                         action="store_const", const=False,
                         help="Only include subjects who don't wear glasses.")
    p.set_defaults(wears_glasses=cfg.filter.wears_glasses)

    # Consumer launch mode. Default: spawn the real consumer subprocess.
    # --no-consumer leaves the provider running alone (e.g. when the consumer
    # runs on another host).
    p.add_argument(
        "--no-consumer", dest="consumer_mode",
        action="store_const", const="none",
        help="Provider-only mode; don't launch any consumer.",
    )
    p.set_defaults(consumer_mode=cfg.runtime.consumer_mode)
    p.add_argument(
        "--consumer-cmd", default=cfg.runtime.consumer_cmd,
        help=(
            "Custom command to run the consumer as a subprocess. Overrides the "
            "default `python -u consumer.py PORT [PORT ...]` invocation."
        ),
    )

    p.add_argument("--seed", type=int, default=cfg.runtime.seed,
                   help="RNG seed for chunk shuffling. Default: nondeterministic.")
    p.add_argument("--log-level", default=cfg.runtime.log_level,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def _run_subprocess_consumer(cmd: str) -> int:
    """Spawn a real consumer command and propagate its exit code."""
    log = logging.getLogger("orchestrator")
    log.info("launching consumer subprocess: %s", cmd)
    proc = await asyncio.create_subprocess_exec(*shlex.split(cmd))
    return await proc.wait()


async def _orchestrate(args: argparse.Namespace) -> int:
    log = logging.getLogger("orchestrator")

    # 0. Preflight: verify data dir is populated; build visit_db.json from the
    #    per-visit JSON sidecars so the consumer's UUID-hex lookup hits.
    check_data_dir(args.data_dir)
    build_visit_db(args.data_dir, Path("visit_db.json"))

    # 1. Load and filter visits.
    visits = load_all_visits(args.data_dir)
    log.info("loaded %d visits from %s", len(visits), args.data_dir)

    # Merge the filter: CLI flags override config.yaml's filter section per field.
    # An unset CLI flag stays as None (the argparse default we configured), and
    # we fall back to the config value for that field.
    cfg = load_config()
    visit_filter = build_filter(
        genders        = args.gender        if args.gender        is not None else list(cfg.filter.gender),
        ages           = args.age           if args.age           is not None else list(cfg.filter.age),
        names          = args.name          if args.name          is not None else list(cfg.filter.names),
        visit_ids      = args.visit_id      if args.visit_id      is not None else list(cfg.filter.visit_ids),
        person_ids     = args.person_id     if args.person_id     is not None else list(cfg.filter.person_ids),
        dominant_hands = args.dominant_hand if args.dominant_hand is not None else list(cfg.filter.dominant_hand),
        wears_glasses  = args.wears_glasses,
    )
    log.info("applying filter: %s", visit_filter.describe())
    matching = visit_filter.apply(visits)
    log.info("%d/%d visits matched", len(matching), len(visits))
    if not matching:
        log.error("no visits match the filter; nothing to train on")
        return 2

    # 2. Plan and shard chunks.
    rng = random.Random(args.seed)
    chunks = plan_chunks(matching, rng=rng)
    if not chunks:
        log.error("matched visits produced zero chunks (files too short?)")
        return 2

    shards = shard_round_robin(chunks, num_shards=len(args.ports))
    log.info(
        "planned %d chunks across %d ports (per-port: %s)",
        len(chunks), len(args.ports), [len(s) for s in shards],
    )

    # 3. Run provider; concurrently run consumer once ports are up.
    provider_ready = asyncio.Event()
    provider_task = asyncio.create_task(
        run_provider(shards, args.ports, host=args.host, ready=provider_ready),
        name="provider",
    )

    consumer_task: asyncio.Task | None = None
    await provider_ready.wait()

    if args.consumer_mode == "none":
        log.info("provider-only mode; waiting for external consumer on ports %s",
                 args.ports)
    else:  # "real" (default)
        cmd = args.consumer_cmd or (
            f"python -u consumer.py {' '.join(str(p) for p in args.ports)}"
        )
        consumer_task = asyncio.create_task(
            _run_subprocess_consumer(cmd),
            name="consumer-subprocess",
        )

    # SIGINT/SIGTERM cancels both cleanly.
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass  # Windows

    tasks_to_wait: list[asyncio.Task] = [provider_task]
    if consumer_task is not None:
        tasks_to_wait.append(consumer_task)

    stop_waiter = asyncio.create_task(stop.wait())
    try:
        done, _pending = await asyncio.wait(
            tasks_to_wait + [stop_waiter],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if stop_waiter in done:
            log.info("signal received; shutting down")
            for t in tasks_to_wait:
                t.cancel()
        else:
            stop_waiter.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_wait, return_exceptions=True),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                for t in tasks_to_wait:
                    if not t.done():
                        t.cancel()
    finally:
        for t in tasks_to_wait + [stop_waiter]:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks_to_wait, return_exceptions=True)

    # Data-integrity summary: provider returns a StreamStats per port, so we
    # can tell whether every planned chunk made it onto the wire. Anything less
    # than 100% per port indicates the consumer disconnected mid-stream.
    if provider_task.done() and not provider_task.cancelled():
        try:
            stats = provider_task.result()
        except Exception:
            stats = []
        if stats:
            total_planned = sum(s.chunks_planned for s in stats)
            total_streamed = sum(s.chunks_streamed for s in stats)
            total_bytes = sum(s.bytes_streamed for s in stats)
            all_clean = all(s.is_complete for s in stats)

            if all_clean:
                log.info(
                    "integrity OK: %d/%d chunks streamed across %d ports (%.1f MB total)",
                    total_streamed, total_planned, len(stats),
                    total_bytes / (1024 * 1024),
                )
            else:
                log.warning(
                    "integrity FAIL: %d/%d chunks streamed across %d ports (%d lost)",
                    total_streamed, total_planned, len(stats),
                    total_planned - total_streamed,
                )
                for s in stats:
                    if not s.is_complete:
                        log.warning(
                            "  port %d: %d/%d chunks streamed (%d lost, clean_close=%s)",
                            s.port, s.chunks_streamed, s.chunks_planned,
                            s.chunks_lost, s.clean_close,
                        )
                return 2  # non-zero exit so CI can flag data loss

    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    _setup_logging(args.log_level)
    try:
        return asyncio.run(_orchestrate(args))
    except KeyboardInterrupt:
        return 130
    except FileNotFoundError as e:
        # Preflight errors carry a multi-line guidance message in `str(e)`.
        # Print it cleanly without a traceback so the reviewer sees actionable
        # text instead of a wall of Python internals.
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
