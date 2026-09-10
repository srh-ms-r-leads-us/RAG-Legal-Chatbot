import json
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import pdf_inspector
import pdfplumber
from pdfminer.high_level import extract_text_to_fp

# ============================================================
# Configuration
# ============================================================

PDF_PATH = Path("input/ECE_PB29_EN.pdf")
OUTPUT_ROOT = Path("output")

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".jp2",
    ".jbig2",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# Utilities
# ============================================================

def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def prepare_output(name: str):
    root = OUTPUT_ROOT / name
    reset_dir(root)

    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)

    return root, images


def write_meta(
        root: Path,
        *,
        tool: str,
        elapsed: float,
        success: bool,
        error: str | None = None,
        **kwargs,
):
    metadata = {
        "tool": tool,
        "input": str(PDF_PATH),
        "elapsed_seconds": round(elapsed, 3),
        "success": success,
        "error": error,
        **kwargs,
    }

    with open(root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )


def copy_images(source_root: Path, target_dir: Path):
    """
    Recursively collect images produced by a tool.
    """

    count = 0

    for path in source_root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # Don't copy files already inside target_dir
        if target_dir in path.parents:
            continue

        target = target_dir / path.name

        # avoid filename collision
        if target.exists():
            stem = target.stem
            suffix = target.suffix

            i = 1

            while target.exists():
                target = target_dir / f"{stem}_{i}{suffix}"
                i += 1

        shutil.copy2(path, target)
        count += 1

    return count


# ============================================================
# 1. PDFMiner
# ============================================================

def run_pdfminer():
    name = "pdfminer"
    root, images_dir = prepare_output(name)

    start = time.perf_counter()

    try:
        output_file = root / "result.md"

        # pdfminer can extract text and embedded raster images.
        with open(PDF_PATH, "rb") as fin:
            with open(
                    output_file,
                    "w",
                    encoding="utf-8",
            ) as fout:
                extract_text_to_fp(
                    fin,
                    fout,
                    output_type="text",
                    codec="utf-8",
                    output_dir=str(images_dir),
                )

        elapsed = time.perf_counter() - start

        image_count = len(
            [
                p
                for p in images_dir.iterdir()
                if p.is_file()
            ]
        )

        write_meta(
            root,
            tool=name,
            elapsed=elapsed,
            success=True,
            image_count=image_count,
        )

        return True

    except Exception:
        elapsed = time.perf_counter() - start
        error = traceback.format_exc()

        write_meta(
            root,
            tool=name,
            elapsed=elapsed,
            success=False,
            error=error,
        )

        return False


# ============================================================
# 2. PDFPlumber
# ============================================================

def run_pdfplumber():
    name = "pdfplumber"
    root, images_dir = prepare_output(name)

    start = time.perf_counter()

    try:
        pages = []

        with pdfplumber.open(PDF_PATH) as pdf:

            for page_number, page in enumerate(
                    pdf.pages,
                    start=1,
            ):
                text = page.extract_text(
                    layout=True,
                )

                pages.append(
                    f"<!-- Page {page_number} -->\n\n"
                    + (text or "")
                )

        markdown = "\n\n".join(pages)

        (root / "result.md").write_text(
            markdown,
            encoding="utf-8",
        )

        # Important:
        # pdfplumber exposes image objects, but does not provide
        # a general high-level "export all images as PNG/JPEG"
        # equivalent to Marker.
        #
        # Leave images/ empty intentionally.
        #
        # This is part of what the benchmark is testing.

        (images_dir / "README.txt").write_text(
            "pdfplumber exposes page.images metadata, "
            "but this benchmark does not add a separate image "
            "decoding implementation because that would introduce "
            "custom post-processing into the comparison.\n",
            encoding="utf-8",
        )

        elapsed = time.perf_counter() - start

        write_meta(
            root,
            tool=name,
            elapsed=elapsed,
            success=True,
            image_count=0,
            note=(
                "result.md contains layout-preserved extracted text. "
                "No additional Markdown reconstruction was applied."
            ),
        )

        return True

    except Exception:
        elapsed = time.perf_counter() - start
        error = traceback.format_exc()

        write_meta(
            root,
            tool=name,
            elapsed=elapsed,
            success=False,
            error=error,
        )

        return False


# ============================================================
# 3. PDF Inspector
# ============================================================

def run_pdf_inspector():
    name = "pdf-inspector"
    root, images_dir = prepare_output(name)

    start = time.perf_counter()

    try:
        result = pdf_inspector.process_pdf(
            str(PDF_PATH)
        )

        markdown = result.markdown or ""

        (root / "result.md").write_text(
            markdown,
            encoding="utf-8",
        )

        # pdf-inspector's public high-level API returns Markdown
        # but does not currently expose Marker-style extracted
        # image files.
        (images_dir / "README.txt").write_text(
            "pdf-inspector Markdown extraction does not export "
            "standalone image files through process_pdf().\n",
            encoding="utf-8",
        )

        elapsed = time.perf_counter() - start

        write_meta(
            root,
            tool=name,
            elapsed=elapsed,
            success=True,
            image_count=0,
            pdf_type=str(result.pdf_type),
        )

        return True

    except Exception:
        elapsed = time.perf_counter() - start
        error = traceback.format_exc()

        write_meta(
            root,
            tool=name,
            elapsed=elapsed,
            success=False,
            error=error,
        )

        return False


# ============================================================
# Marker common runner
# ============================================================

def run_marker_command(
        name: str,
        extra_args: list[str] | None = None,
):
    root, images_dir = prepare_output(name)

    raw_dir = root / "_raw"
    raw_dir.mkdir()

    start = time.perf_counter()

    try:
        command = [
            "marker_single",
            str(PDF_PATH),
            "--output_dir",
            str(raw_dir),
            "--output_format",
            "markdown",
            "--paginate_output",
        ]

        if extra_args:
            command.extend(extra_args)

        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        elapsed = time.perf_counter() - start

        (root / "stdout.log").write_text(
            process.stdout or "",
            encoding="utf-8",
        )

        (root / "stderr.log").write_text(
            process.stderr or "",
            encoding="utf-8",
        )

        if process.returncode != 0:
            raise RuntimeError(
                f"Marker exited with code "
                f"{process.returncode}\n\n"
                f"{process.stderr}"
            )

        # Marker normally creates its own nested output directory.
        md_files = list(raw_dir.rglob("*.md"))

        if not md_files:
            raise RuntimeError(
                "Marker finished successfully but "
                "no Markdown file was found."
            )

        # Usually only one PDF is being converted here.
        marker_md = md_files[0]

        shutil.copy2(
            marker_md,
            root / "result.md",
        )

        image_count = copy_images(
            raw_dir,
            images_dir,
        )

        write_meta(
            root,
            tool=name,
            elapsed=elapsed,
            success=True,
            command=command,
            image_count=image_count,
        )

        return True

    except Exception:
        elapsed = time.perf_counter() - start
        error = traceback.format_exc()

        write_meta(
            root,
            tool=name,
            elapsed=elapsed,
            success=False,
            error=error,
        )

        return False


# ============================================================
# 4. Marker normal
# ============================================================

def run_marker():
    return run_marker_command(
        "marker",
        [
            "--disable_ocr"
        ]
    )


# ============================================================
# 5. Marker force OCR
# ============================================================

def run_marker_ocr():
    return run_marker_command(
        "marker-ocr",
        [
            "--mode",
            "fast",
            "--force_ocr",
            "--strip_existing_ocr",
        ],
    )


# ============================================================
# Main
# ============================================================

RUNNERS = [
    ("pdfminer", run_pdfminer),
    ("pdfplumber", run_pdfplumber),
    ("pdf-inspector", run_pdf_inspector),
    ("marker", run_marker),
    ("marker-ocr", run_marker_ocr),
]


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("PDF -> Markdown benchmark")
    print("=" * 70)
    print(f"Input: {PDF_PATH}")
    print()

    results = {}

    for name, runner in RUNNERS:
        print(f"[RUN] {name}")

        try:
            success = runner()
        except Exception:
            success = False
            traceback.print_exc()

        results[name] = success

        status = "OK" if success else "FAILED"

        print(f"[{status}] {name}")
        print()

    print("=" * 70)
    print("Summary")
    print("=" * 70)

    for name, success in results.items():
        print(
            f"{name:<20} "
            f"{'OK' if success else 'FAILED'}"
        )

    print()
    print(f"Results: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
