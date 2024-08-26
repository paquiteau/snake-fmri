"""CLI for SNAKE."""

import logging
import hydra

from omegaconf import OmegaConf
from snake.core.engine import BaseAcquisitionEngine
from snake.core.handlers import HandlerList
from snake.mrd_utils import make_base_mrd
from snake.core.phantom import Phantom
from snake.core.smaps import get_smaps
from snake_toolkit.cli.config import ConfigSNAKE, cleanup_cuda

log = logging.getLogger(__name__)


def acquisition(cfg: ConfigSNAKE) -> None:
    """Simulate acquisition."""
    cfg: ConfigSNAKE = OmegaConf.to_object(cfg)
    # FIXME: Hydra should be able to do that on its own.
    print(cfg)
    sim_conf = cfg.sim_conf
    if cfg.phantom.name == "brainweb":
        phantom = Phantom.from_brainweb(
            sub_id=cfg.phantom.sub_id,
            sim_conf=sim_conf,
            tissue_select=cfg.phantom.tissue_select,
            tissue_ignore=cfg.phantom.tissue_ignore,
            tissue_file=cfg.phantom.tissue_file,
        )
    else:
        raise ValueError(f"Unknown phantom {cfg.phantom.name}")

    handlers = HandlerList(*cfg.handlers.values())

    if len(cfg.sampler) > 1:
        log.warning(
            "Multiple sampler configuration detected. Only the first one is used."
        )
    sampler = list(cfg.sampler.values())[0]

    for h in handlers:
        phantom = h.get_static(phantom, sim_conf)

    dynamic_data = [h.get_dynamic(phantom, sim_conf) for h in handlers]

    smaps = None
    if sim_conf.hardware.n_coils > 1:
        smaps = get_smaps(sim_conf.shape, n_coils=sim_conf.hardware.n_coils)

    make_base_mrd(cfg.filename, sampler, phantom, sim_conf, dynamic_data, smaps)
    log.info("Initialization done")

    engine_klass = BaseAcquisitionEngine.__registry__[sampler.__engine__]
    kwargs = {}
    if engine_klass.__engine_name__ == "NUFFT":
        kwargs["nufft_backend"] = cfg.engine.nufft_backend
    engine = engine_klass(model=cfg.engine.model, snr=cfg.engine.snr)

    engine(
        cfg.filename,
        worker_chunk_size=cfg.engine.chunk_size,
        n_workers=cfg.engine.n_jobs,
        **kwargs,
    )

    log.info("Acquisition done")
    log.info("Output file is at %s", cfg.filename)
    cleanup_cuda()


acquisition_cli = hydra.main(
    version_base=None, config_path="../../cli-conf", config_name="config"
)(acquisition)

if __name__ == "__main__":
    acquisition_cli()
