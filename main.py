import asyncio
import importlib
import json
import logging
from pathlib import Path
import sys

from ratelimitedapi import RateLimitedApi
from coordinator import Coordinator

if __name__ == "__main__":
    # Working dir needs to have config.json file at the root.  Coordinator will create subfolders for storing the
    # backtest results.
    working_dir = Path(r"path/to/data/dir")
    with (working_dir / "config.json").open() as f:
        config = json.load(f)

    loglevel = config.get("loglevel", logging.INFO)
    logging.basicConfig(  # configures root logger, more succinct than wiring up handlers on object directly
        level=loglevel,
        format="%(asctime)s %(name)s [%(threadName)s] [%(levelname)-5.5s]: %(message)s",
        stream=sys.stdout
    )
    logger = logging.getLogger()  # root logger
    logger.info(str(config))

    try:
        mod = importlib.import_module(config["module"])
        testset = getattr(mod, config["testset"])
        api = RateLimitedApi(config["user_id"], config["token"], debug=False)
        coordinator = Coordinator(testset, api, working_dir, config["project_name"], config["concurrency"])
        asyncio.run(coordinator.run(), debug=True)
    except Exception as e:
        logger.error("Unhandled error", exc_info=e)
        sys.exit(1)
