#!/usr/bin/env python3
"""
Generate basic statistics for NYC Tax Law documents.

This script analyzes the raw document files and outputs statistics including:
- Line count, word count, character count
- Token estimates (heuristic and tiktoken-based)
- Document structure analysis (chapters, subchapters, sections)
"""

import re
import sys
from datetime import datetime
from pathlib import Path

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


def count_basic_stats(text: str) -> dict:
    """Count basic text statistics."""
    lines = text.split('\n')
    words = text.split()

    return {
        'lines': len(lines),
        'words': len(words),
        'characters': len(text),
        'characters_no_whitespace': len(text.replace(' ', '').replace('\n', '').replace('\t', '')),
    }


def estimate_tokens(text: str) -> dict:
    """Estimate token counts using various methods."""
    estimates = {}

    # Heuristic: ~4 characters per token (common approximation)
    estimates['heuristic_4_chars'] = len(text) // 4

    # Heuristic: ~0.75 tokens per word (common for English)
    word_count = len(text.split())
    estimates['heuristic_words'] = int(word_count * 0.75)

    # Tiktoken (if available) - use cl100k_base (GPT-4/Claude tokenizer approximation)
    if TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.get_encoding('cl100k_base')
            estimates['tiktoken_cl100k'] = len(enc.encode(text))
        except Exception as e:
            estimates['tiktoken_error'] = str(e)

    return estimates


def analyze_structure(text: str) -> dict:
    """Analyze document structure (chapters, subchapters, sections)."""
    structure = {}

    # Count chapters (pattern: "Chapter N:" or "CHAPTER N")
    chapter_pattern = r'Chapter\s+\d+[:\s]'
    chapters = re.findall(chapter_pattern, text, re.IGNORECASE)
    structure['chapters'] = len(set(chapters))

    # Count subchapters (pattern: "Subchapter N:")
    subchapter_pattern = r'Subchapter\s+\d+[:\s]'
    subchapters = re.findall(subchapter_pattern, text, re.IGNORECASE)
    structure['subchapters'] = len(subchapters)

    # Count sections (pattern: "§ 11-XXX" or "Section 11-XXX")
    section_pattern = r'§\s*11-\d+'
    sections = re.findall(section_pattern, text)
    unique_sections = set(sections)
    structure['sections'] = len(unique_sections)
    structure['section_references'] = len(sections)

    # Extract section range
    if unique_sections:
        section_numbers = [int(re.search(r'11-(\d+)', s).group(1)) for s in unique_sections]
        structure['section_range'] = f"11-{min(section_numbers)} to 11-{max(section_numbers)}"

    return structure


def generate_markdown_report(
    filename: str,
    basic_stats: dict,
    token_estimates: dict,
    structure: dict,
) -> str:
    """Generate markdown report."""

    report = f"""# NYC Tax Law Document Statistics

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Source Document

| Property | Value |
|----------|-------|
| Filename | `{filename}` |
| Format | {'HTML' if filename.endswith('.html') else 'Plain Text'} |

---

## Basic Statistics

| Metric | Value |
|--------|-------|
| Lines | {basic_stats['lines']:,} |
| Words | {basic_stats['words']:,} |
| Characters (total) | {basic_stats['characters']:,} |
| Characters (no whitespace) | {basic_stats['characters_no_whitespace']:,} |

---

## Token Estimates

Understanding token counts is critical for LLM context window planning.

| Method | Estimated Tokens | Notes |
|--------|-----------------|-------|
| Heuristic (~4 chars/token) | ~{token_estimates['heuristic_4_chars']:,} | Simple approximation |
| Heuristic (~0.75 tokens/word) | ~{token_estimates['heuristic_words']:,} | Word-based estimate |
"""

    if 'tiktoken_cl100k' in token_estimates:
        report += f"| tiktoken (cl100k_base) | {token_estimates['tiktoken_cl100k']:,} | Accurate GPT-4/Claude estimate |\n"
    elif 'tiktoken_error' in token_estimates:
        report += f"| tiktoken | Error | {token_estimates['tiktoken_error']} |\n"
    else:
        report += "| tiktoken | N/A | Install tiktoken for accurate count |\n"

    report += f"""
### Context Window Analysis

Based on token estimates, this document:
- **Exceeds** Claude's standard 200K context window: {'Yes' if token_estimates.get('tiktoken_cl100k', token_estimates['heuristic_4_chars']) > 200000 else 'No'}
- **Requires chunking** for RAG: Yes (document too large for single context)
- **Recommended approach**: Section-aware chunking with ~800 token chunks

---

## Document Structure

| Element | Count |
|---------|-------|
| Chapters | {structure.get('chapters', 'N/A')} |
| Subchapters | {structure.get('subchapters', 'N/A')} |
| Unique Sections | {structure.get('sections', 'N/A')} |
| Section References | {structure.get('section_references', 'N/A')} |
| Section Range | {structure.get('section_range', 'N/A')} |

### Structure Notes

- Document follows NYC Administrative Code Title 11 structure
- Sections follow pattern: `§ 11-XXX` (e.g., § 11-201, § 11-322)
- Hierarchical: Title → Chapter → Subchapter → Section → Subsection
- Subsections use lettered format: (a), (b), (c), etc.
- Nested items use numbered format: (1), (2), (3), etc.

---

## Recommendations for RAG Implementation

1. **Chunking Strategy**: Use section-aware chunking that respects `§ 11-XXX` boundaries
2. **Metadata**: Preserve chapter/subchapter hierarchy in chunk metadata
3. **Overlap**: Use 100-200 token overlap to maintain context
4. **Fallback**: Generate compressed prompt (~50K tokens) for simple queries
5. **Embeddings**: Use text-embedding-3-small for cost-effective vector search

---

*This report was auto-generated by `scripts/generate_basic_stats.py`*
"""

    return report


def main():
    """Main entry point."""
    # Determine project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Default to TXT file for analysis (cleaner for text processing)
    raw_dir = project_root / 'data' / 'raw'
    txt_file = raw_dir / 'New York City Finance Chapter 1.txt'
    html_file = raw_dir / 'New York City Finance Chapter 1.html'

    # Choose file to analyze
    if txt_file.exists():
        input_file = txt_file
    elif html_file.exists():
        input_file = html_file
    else:
        print(f"Error: No source files found in {raw_dir}")
        print("Expected: 'New York City Finance Chapter 1.txt' or '.html'")
        sys.exit(1)

    print(f"Analyzing: {input_file.name}")
    print("-" * 50)

    # Read file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Generate statistics
    print("Computing basic statistics...")
    basic_stats = count_basic_stats(text)

    print("Estimating token counts...")
    token_estimates = estimate_tokens(text)

    print("Analyzing document structure...")
    structure = analyze_structure(text)

    # Print summary to console
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Lines:      {basic_stats['lines']:,}")
    print(f"Words:      {basic_stats['words']:,}")
    print(f"Characters: {basic_stats['characters']:,}")
    print(f"Tokens (est): ~{token_estimates.get('tiktoken_cl100k', token_estimates['heuristic_4_chars']):,}")
    print(f"Sections:   {structure.get('sections', 'N/A')}")
    print(f"Chapters:   {structure.get('chapters', 'N/A')}")
    print("=" * 50)

    # Generate markdown report
    report = generate_markdown_report(
        filename=input_file.name,
        basic_stats=basic_stats,
        token_estimates=token_estimates,
        structure=structure,
    )

    # Write report
    output_dir = project_root / 'data' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'basic_stats.md'

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nReport saved to: {output_file}")

    # Also analyze HTML if both exist
    if txt_file.exists() and html_file.exists() and input_file == txt_file:
        print("\n" + "-" * 50)
        print(f"Also analyzing: {html_file.name}")

        with open(html_file, 'r', encoding='utf-8') as f:
            html_text = f.read()

        html_stats = count_basic_stats(html_text)
        html_tokens = estimate_tokens(html_text)

        print(f"HTML Lines:      {html_stats['lines']:,}")
        print(f"HTML Characters: {html_stats['characters']:,}")
        print(f"HTML Tokens (est): ~{html_tokens.get('tiktoken_cl100k', html_tokens['heuristic_4_chars']):,}")


if __name__ == '__main__':
    main()
