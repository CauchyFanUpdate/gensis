#!/usr/bin/env python3
"""
Generate gemini-based stage1 and stage2 scripts for every pipeline in gen_banana_list.json.

For each pipeline directory, reads the existing generate_samples_v5_stage1.py /
generate_samples_v5_stage2.py (or stage1_edit.py / stage2_diversify.py for mmmu_pro),
applies targeted substitutions to replace GPT-Image-2 with Gemini, and writes:
  generate_samples_gemini_stage1.py
  generate_samples_gemini_stage2.py

Also copies _gemini_image.py into each pipeline directory so scripts can import it.

Run once from the ZeroGeneration root:
  python create_gemini_scripts.py
"""

import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
BANANA = ROOT / "gen_banana_list.json"
GEMINI_IMAGE_SRC = ROOT / "_gemini_image.py"

# ── Substitution rules ────────────────────────────────────────────────────

def _apply_substitutions(src: str, stage: int) -> str:
    """Apply all GPT→Gemini substitutions to a script's source text."""

    # 1. Remove gpt_image_edit from the top-level import block only.
    #    Split src into import section + rest to avoid matching local imports inside functions.
    split_idx = _find_import_section_end(src)
    import_section = src[:split_idx]
    rest = src[split_idx:]
    # Remove ", gpt_image_edit," / "gpt_image_edit," / ", gpt_image_edit" / "gpt_image_edit"
    # from the import section cleanly (handles commas and surrounding whitespace).
    import_section = re.sub(r',\s*\bgpt_image_edit\b', '', import_section)
    import_section = re.sub(r'\bgpt_image_edit\b\s*,\s*', '', import_section)
    import_section = re.sub(r'\bgpt_image_edit\b', '', import_section)
    src = import_section + rest

    # 2. Insert gemini_image_edit import after the _v5_common/_common import block
    src = _inject_gemini_import(src)

    # 3. Replace gpt_image_edit( call with gemini_image_edit( and swap proxy→inner
    #    Pattern handles multi-line: gpt_image_edit(\n    self._proxy, ...
    src = re.sub(
        r'\bawait gpt_image_edit\(\s*self\._proxy,\s*self\.gpt_image_endpoint,\s*self\.gpt_image_api_key,',
        'await gemini_image_edit(\n                self._inner, self.gemini_image_endpoint, self.gemini_image_api_key,',
        src,
    )
    # Fallback: any remaining gpt_image_edit( call
    src = src.replace('await gpt_image_edit(', 'await gemini_image_edit(')

    # 4. Replace references to module-level constants (mmmu_pro style)
    src = src.replace('GPT_IMAGE_ENDPOINT', 'GEMINI_IMAGE_ENDPOINT')
    src = src.replace('GPT_IMAGE_API_KEY', 'GEMINI_IMAGE_API_KEY')

    # 5. Class __init__ parameter names
    src = src.replace('gpt_image_endpoint:', 'gemini_image_endpoint:')
    src = src.replace('gpt_image_api_key:', 'gemini_image_api_key:')
    src = src.replace('self.gpt_image_endpoint', 'self.gemini_image_endpoint')
    src = src.replace('self.gpt_image_api_key', 'self.gemini_image_api_key')
    src = src.replace('gpt_image_endpoint=', 'gemini_image_endpoint=')
    src = src.replace('gpt_image_api_key=', 'gemini_image_api_key=')
    # local variable names inside functions
    src = src.replace('gpt_image_endpoint', 'gemini_image_endpoint')
    src = src.replace('gpt_image_api_key', 'gemini_image_api_key')

    # 6. Error reason labels in jsonl records
    src = src.replace('"call3a_gpt_image_failed"', '"call3a_gemini_image_failed"')
    src = src.replace('"call3a_no_image"',          '"call3a_gemini_no_image"')
    src = src.replace('"call3b_gpt_image_failed"', '"call3b_gemini_image_failed"')
    src = src.replace('"call3b_no_image"',          '"call3b_gemini_no_image"')
    src = src.replace('"call3_gpt_image_failed"',  '"call3_gemini_image_failed"')
    src = src.replace('"call3_no_image"',           '"call3_gemini_no_image"')

    # 7. Log prefixes in the call (e.g. log_prefix=f"gpt3a/...")
    src = src.replace('log_prefix=f"gpt3a/', 'log_prefix=f"gemini3a/')
    src = src.replace('log_prefix=f"gpt3b/', 'log_prefix=f"gemini3b/')
    src = src.replace("log_prefix=f'gpt3a/", "log_prefix=f'gemini3a/")
    src = src.replace("log_prefix=f'gpt3b/", "log_prefix=f'gemini3b/")

    # 8. CLI argument strings
    src = src.replace('"--gpt_image_endpoint"', '"--gemini_image_endpoint"')
    src = src.replace('"--gpt_image_api_key"',  '"--gemini_image_api_key"')
    src = src.replace("'--gpt_image_endpoint'", "'--gemini_image_endpoint'")
    src = src.replace("'--gpt_image_api_key'",  "'--gemini_image_api_key'")

    # 9. Validation messages
    src = src.replace('--gpt_image_api_key / GPT_IMAGE_API_KEY', '--gemini_image_api_key / GEMINI_IMAGE_API_KEY')
    src = src.replace('--gpt_image_endpoint / GPT_IMAGE_ENDPOINT', '--gemini_image_endpoint / GEMINI_IMAGE_ENDPOINT')

    # 10. Update output dir default: _archive/output_v5 → _archive/output_gemini
    src = re.sub(
        r'(default\s*=\s*["\'])([^"\']*_archive/output_v5)(["\'])',
        lambda m: m.group(1) + m.group(2).replace('output_v5', 'output_gemini') + m.group(3),
        src,
    )
    # Also handle the zerobench_pipeline/ prefix variant
    src = re.sub(
        r'(default\s*=\s*["\'])zerobench_pipeline/_archive/output_v5(["\'])',
        lambda m: m.group(1) + '_archive/output_gemini' + m.group(2),
        src,
    )

    # 11. Update default concurrency and retries in argparse
    # Replace the env-var string default inside os.environ.get("MAX_CONCURRENT", "16") → "64"
    src = re.sub(
        r'(os\.environ\.get\(["\']MAX_CONCURRENT["\'],\s*["\'])(\d+)(["\'])',
        lambda m: m.group(1) + '64' + m.group(3),
        src,
    )
    src = re.sub(
        r'(os\.environ\.get\(["\']MAX_RETRIES["\'],\s*["\'])(\d+)(["\'])',
        lambda m: m.group(1) + '20' + m.group(3),
        src,
    )
    # Also fix class-level default value / inline defaults
    src = re.sub(
        r'(max_concurrent\s*:\s*int\s*=\s*)16(\b)',
        r'\g<1>64\2',
        src,
    )
    src = re.sub(
        r'(max_retries\s*:\s*int\s*=\s*)6(\b)',
        r'\g<1>20\2',
        src,
    )

    # 12. Replace API key default values in argparse / env defaults with new key
    GEMINI_KEY = "YOUR_GEMINI_IMAGE_API_KEY"
    src = re.sub(
        r'(add_argument\(["\']--gemini_image_api_key["\'].*?default=os\.environ\.get\(["\']GEMINI_IMAGE_API_KEY["\'],\s*["\'])[^"\']*(["\'])',
        lambda m: m.group(1) + GEMINI_KEY + m.group(2),
        src,
    )
    # Simpler pattern: default="" for the api_key arg
    src = re.sub(
        r'(add_argument\(["\']--gemini_image_api_key["\'].*?default=)["\']["\']',
        lambda m: m.group(1) + f'"{GEMINI_KEY}"',
        src,
    )

    # 13. Update endpoint default to Gemini image endpoint
    GEMINI_ENDPOINT = "https://runway.devops.rednote.life/openai/google/v1:generateContent"
    src = re.sub(
        r'(add_argument\(["\']--gemini_image_endpoint["\'].*?default=os\.environ\.get\(["\']GEMINI_IMAGE_ENDPOINT["\'],\s*["\'])[^"\']*(["\'])',
        lambda m: m.group(1) + GEMINI_ENDPOINT + m.group(2),
        src,
    )
    src = re.sub(
        r'(add_argument\(["\']--gemini_image_endpoint["\'].*?default=)["\'][^"\']*["\']',
        lambda m: m.group(1) + f'"{GEMINI_ENDPOINT}"',
        src,
    )

    # 14. Update docstring header
    src = _update_docstring(src, stage)

    return src


def _find_import_section_end(src: str) -> int:
    """Return the character index where the top-level import block ends.

    Scans line by line; tracks parenthesis depth for multi-line imports; stops
    at the first top-level non-import, non-blank, non-comment line.
    """
    lines = src.split('\n')
    last_import_line = 0   # line index (0-based)
    paren_depth = 0
    found_first_import = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        is_top_level = not (line.startswith(' ') or line.startswith('\t'))

        if paren_depth > 0:
            paren_depth += line.count('(') - line.count(')')
            last_import_line = i
            continue

        if not stripped or stripped.startswith('#'):
            continue

        # Module-level docstring at the very start
        if i < 5 and (stripped.startswith('"""') or stripped.startswith("'''")):
            continue

        if is_top_level and (stripped.startswith('from ') or stripped.startswith('import ')):
            found_first_import = True
            last_import_line = i
            depth = line.count('(') - line.count(')')
            if depth > 0:
                paren_depth = depth
        elif found_first_import and is_top_level:
            break

    # Convert line index → character offset (include the entire last import line + newline)
    char_offset = 0
    for i, line in enumerate(lines):
        char_offset += len(line) + 1  # +1 for '\n'
        if i == last_import_line:
            break
    return min(char_offset, len(src))


def _inject_gemini_import(src: str) -> str:
    """Add import of gemini_image_edit from _gemini_image after the top-level import block."""
    import_block = (
        '\nimport sys as _sys\n'
        '_sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))\n'
        'from _gemini_image import (\n'
        '    gemini_image_edit,\n'
        '    GEMINI_IMAGE_ENDPOINT,\n'
        '    GEMINI_IMAGE_API_KEY,\n'
        ')\n'
    )
    split_idx = _find_import_section_end(src)
    if split_idx > 0:
        return src[:split_idx] + import_block + src[split_idx:]
    return import_block + src


def _update_docstring(src: str, stage: int) -> str:
    """Update the module docstring to mention Gemini."""
    model_label = 'gemini-3-pro-image-preview'
    old_labels = ['GPT-Image-2', 'gpt-image-2', 'GPT Image 2']
    for lbl in old_labels:
        src = src.replace(lbl, model_label)
    return src


# ── mmmu_pro: _common.py uses different constant names ───────────────────

def _apply_mmmu_substitutions(src: str) -> str:
    """_common.py in mmmu_pro defines GPT_IMAGE_ENDPOINT/API_KEY at module level."""
    GEMINI_KEY = "YOUR_GEMINI_IMAGE_API_KEY"
    GEMINI_ENDPOINT = "https://runway.devops.rednote.life/openai/google/v1:generateContent"
    # Replace the constant definitions
    src = re.sub(
        r"GPT_IMAGE_ENDPOINT\s*=\s*os\.environ\.get\([^)]+\)",
        f'GEMINI_IMAGE_ENDPOINT = os.environ.get("GEMINI_IMAGE_ENDPOINT", "{GEMINI_ENDPOINT}")',
        src,
    )
    src = re.sub(
        r"GPT_IMAGE_API_KEY\s*=\s*os\.environ\.get\([^)]+\)",
        f'GEMINI_IMAGE_API_KEY = os.environ.get("GEMINI_IMAGE_API_KEY", "{GEMINI_KEY}")',
        src,
    )
    return _apply_substitutions(src, stage=1)


# ── Per-pipeline script discovery ─────────────────────────────────────────

PIPELINE_SCRIPTS = {
    # key: (stage1_src, stage2_src, stage1_dst, stage2_dst)
    # Use None if not applicable
    "default_v5": (
        "generate_samples_v5_stage1.py",
        "generate_samples_v5_stage2.py",
        "generate_samples_gemini_stage1.py",
        "generate_samples_gemini_stage2.py",
    ),
    "mmmu_pro": (
        "stage1_edit.py",
        "stage2_diversify.py",
        "stage1_edit_gemini.py",
        "stage2_diversify_gemini.py",
    ),
}


def _get_script_names(pipeline_dir: Path):
    """Return (stage1_src, stage2_src, stage1_dst, stage2_dst) for a pipeline."""
    if pipeline_dir.name == "gen_pipeline_mmmu_pro":
        return PIPELINE_SCRIPTS["mmmu_pro"]
    return PIPELINE_SCRIPTS["default_v5"]


def _process_pipeline(pipeline_dir: Path) -> None:
    stage1_src, stage2_src, stage1_dst, stage2_dst = _get_script_names(pipeline_dir)

    # Copy _gemini_image.py into the pipeline dir so scripts can import it locally
    dst_gemini_image = pipeline_dir / "_gemini_image.py"
    if not dst_gemini_image.exists() or dst_gemini_image.read_bytes() != GEMINI_IMAGE_SRC.read_bytes():
        shutil.copy2(GEMINI_IMAGE_SRC, dst_gemini_image)
        print(f"  [copy] _gemini_image.py → {pipeline_dir.name}/")

    for src_name, dst_name, stage in [
        (stage1_src, stage1_dst, 1),
        (stage2_src, stage2_dst, 2),
    ]:
        src_path = pipeline_dir / src_name
        dst_path = pipeline_dir / dst_name

        if not src_path.exists():
            print(f"  [skip] {pipeline_dir.name}/{src_name} — not found")
            continue

        original = src_path.read_text(encoding="utf-8")
        modified = _apply_substitutions(original, stage)

        dst_path.write_text(modified, encoding="utf-8")
        print(f"  [OK]   {pipeline_dir.name}/{dst_name}")


def main() -> None:
    if not BANANA.exists():
        raise FileNotFoundError(f"Banana list not found: {BANANA}")

    banana = json.loads(BANANA.read_text(encoding="utf-8"))
    print(f"Processing {len(banana)} pipelines from gen_banana_list.json\n")

    for name, path_str in banana.items():
        pipeline_dir = Path(path_str)
        if not pipeline_dir.is_dir():
            print(f"[warn] {name}: directory not found: {pipeline_dir}")
            continue
        if pipeline_dir.name == "gen_zerobench_output_v5":
            print(f"[skip] {name}: no stage1/stage2 generate scripts")
            continue
        print(f"[{name}]")
        _process_pipeline(pipeline_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
