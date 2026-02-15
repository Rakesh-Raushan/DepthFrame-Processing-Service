"""CLI entry point for the ingestion pipeline.

Usage: uv run python -m depthframe_processing_service.ingest
or make ingest

Environment variable (all optional, with defaults):
- DPS_DATA_DIR: Directory for input CSV and output files (default: 'data')
- DPS_DB_PATH: Path for SQLite database (default: 'data/image_store.db')
- DPS_CSV_FILENAME: Name of the input CSV file (default: 'Challenge2.csv')
- DPS_TARGET_WIDTH: Target width for resized images (default: 150)
- DPS_INTERPOLATION_METHOD: Interpolation method for resizing (default: 'AREA')
"""

import logging
import sys

from depthframe_processing_service.config import settings
from depthframe_processing_service.ingestion.pipeline import run_ingestion

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    try:
        report = run_ingestion(settings)
        logger.info("Done. %d rows ingested.", report.clean_rows)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception:
        logger.exception("Ingestion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
