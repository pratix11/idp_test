from property_intel.config import get_settings
from property_intel.db.repository import DocumentRepository
from property_intel.db.session import get_engine, get_session_factory, init_db
from property_intel.ingestion.organizer import DatasetOrganizer
from property_intel.logging_setup import configure_logging
from property_intel.parsing.docling_parser import DoclingParser
from property_intel.parsing.markitdown_parser import MarkItDownParser
from property_intel.parsing.router import ParserRouter
from property_intel.pipeline.batch_processor import BatchProcessor, DocumentProcessor


def main() -> None:
    settings = get_settings()
    logger = configure_logging(settings.resolved_log_dir(), settings.log_level)

    engine = get_engine()
    init_db(engine)
    session = get_session_factory(engine)()
    repository = DocumentRepository(session)

    router = ParserRouter(primary=DoclingParser(), fallback=MarkItDownParser())
    organizer = DatasetOrganizer(settings.resolved_data_raw_dir())
    processor = DocumentProcessor(repository, router, settings.resolved_data_processed_dir())
    batch_processor = BatchProcessor(organizer, processor)

    summary = batch_processor.run()
    logger.info(
        "Batch complete: %d total, %d completed, %d failed, %d skipped",
        summary.total,
        summary.completed,
        summary.failed,
        summary.skipped,
    )
    for path, message in summary.errors:
        logger.error("  %s: %s", path, message)

    session.close()


if __name__ == "__main__":
    main()
