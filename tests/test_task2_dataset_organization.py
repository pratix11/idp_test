import pytest

from property_intel.config import get_settings
from property_intel.ingestion.organizer import (
    DatasetOrganizer,
    MissingDocumentsError,
    UnknownCategoryError,
)
from property_intel.metadata.schema import DocumentCategory


def _make_pdf(path, content: bytes = b"%PDF-1.4\n%fake\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


@pytest.fixture
def raw_dir(tmp_path):
    root = tmp_path / "raw"
    _make_pdf(root / "maharera" / "acts" / "act1.pdf", b"%PDF-1.4\nacts content\n")
    _make_pdf(root / "maharera" / "circulars" / "2021" / "circ1.pdf", b"%PDF-1.4\ncirculars content\n")
    _make_pdf(root / "maharera" / "regulations" / "reg1.pdf", b"%PDF-1.4\nregulations content\n")
    return root


def test_categorize_assigns_correct_category(raw_dir) -> None:
    organizer = DatasetOrganizer(raw_dir)
    assert organizer.categorize(raw_dir / "maharera" / "acts" / "act1.pdf") == DocumentCategory.ACTS
    assert (
        organizer.categorize(raw_dir / "maharera" / "circulars" / "2021" / "circ1.pdf")
        == DocumentCategory.CIRCULARS
    )
    assert (
        organizer.categorize(raw_dir / "maharera" / "regulations" / "reg1.pdf")
        == DocumentCategory.REGULATIONS
    )


def test_categorize_unknown_category_raises(tmp_path) -> None:
    bad = tmp_path / "raw" / "maharera" / "not_a_real_category" / "x.pdf"
    _make_pdf(bad)
    organizer = DatasetOrganizer(tmp_path / "raw")
    with pytest.raises(UnknownCategoryError):
        organizer.categorize(bad)


def test_scan_finds_all_known_documents(raw_dir) -> None:
    organizer = DatasetOrganizer(raw_dir)
    documents = organizer.scan()
    assert len(documents) == 3
    categories = {doc.category for doc in documents}
    assert categories == {DocumentCategory.ACTS, DocumentCategory.CIRCULARS, DocumentCategory.REGULATIONS}


def test_scan_skips_uncategorizable_files(raw_dir) -> None:
    _make_pdf(raw_dir / "maharera" / "weird_category" / "x.pdf")
    organizer = DatasetOrganizer(raw_dir)
    documents = organizer.scan()
    assert len(documents) == 3


def test_verify_no_missing_passes_when_all_categorized(raw_dir) -> None:
    DatasetOrganizer(raw_dir).verify_no_missing()


def test_verify_no_missing_raises_when_file_uncategorizable(raw_dir) -> None:
    _make_pdf(raw_dir / "maharera" / "weird_category" / "x.pdf")
    organizer = DatasetOrganizer(raw_dir)
    with pytest.raises(MissingDocumentsError):
        organizer.verify_no_missing()


def test_find_duplicates_detects_identical_content(tmp_path) -> None:
    root = tmp_path / "raw"
    content = b"%PDF-1.4\nidentical content\n"
    _make_pdf(root / "maharera" / "circulars" / "a.pdf", content)
    _make_pdf(root / "maharera" / "regulations" / "b.pdf", content)
    _make_pdf(root / "maharera" / "acts" / "unique.pdf", b"%PDF-1.4\nunique\n")

    organizer = DatasetOrganizer(root)
    duplicates = organizer.find_duplicates()

    assert len(duplicates) == 1
    (paths,) = duplicates.values()
    assert len(paths) == 2
    assert {p.name for p in paths} == {"a.pdf", "b.pdf"}


def test_find_duplicates_empty_when_all_unique(raw_dir) -> None:
    organizer = DatasetOrganizer(raw_dir)
    assert organizer.find_duplicates() == {}


@pytest.mark.integration
def test_real_dataset_has_73_documents() -> None:
    settings = get_settings()
    raw_dir = settings.resolved_data_raw_dir()
    if not raw_dir.exists():
        pytest.skip("real data/raw dataset not present")

    organizer = DatasetOrganizer(raw_dir)
    documents = organizer.scan()
    assert len(documents) == 73


@pytest.mark.integration
def test_real_dataset_flags_known_duplicate() -> None:
    settings = get_settings()
    raw_dir = settings.resolved_data_raw_dir()
    if not raw_dir.exists():
        pytest.skip("real data/raw dataset not present")

    organizer = DatasetOrganizer(raw_dir)
    duplicates = organizer.find_duplicates()

    duplicate_names = {p.name for paths in duplicates.values() for p in paths}
    assert "MahaRERA_RA.pdf" in duplicate_names
