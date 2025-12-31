"""
Few-shot examples for zChunk algorithm on NYC Tax Law.

These examples teach Claude how to properly insert split tokens
in NYC tax law documents.
"""

from .system_prompt import BIG_SPLIT_TOKEN, SMALL_SPLIT_TOKEN

# Shorter aliases for readability
B = BIG_SPLIT_TOKEN  # 段 - big split
S = SMALL_SPLIT_TOKEN  # 顿 - small split


# Example input: A section from NYC Finance Chapter 1
EXAMPLE_INPUT = """§ 11-201 Assessments on real property; general powers of finance department.
The commissioner of finance shall be charged generally with the duty and
responsibility of assessing all real property subject to taxation within the
city.
§ 11-202 Maps and records; surveyor.
The commissioner of finance shall appoint a surveyor who shall make the
necessary surveys and corrections of the block or ward maps, and also make all
new tax maps which may be required.
§ 11-203 Maps and records; tax maps.
   a.   As used in the charter and in the code, the term "tax maps" shall mean
and include the block map of taxes and assessments to the extent that the
territory within the city of New York is or shall be embraced in such map, such
ward or land maps as embrace the remainder of such city, and also such maps as
may be prepared under and pursuant to subdivision d of this section.
   b.   Each separately assessed parcel shall be indicated on the tax maps by a
parcel number or by an identification number. A separate identification number
shall be entered upon the tax maps in such manner as clearly to indicate each
separately assessed parcel of real property not indicated by parcel numbering.
Real property indicated by a single identification number shall be deemed to be
a separately assessed parcel."""


# Example output: Same text with split tokens inserted
EXAMPLE_OUTPUT = f"""{B}§ 11-201 Assessments on real property; general powers of finance department.
{S}The commissioner of finance shall be charged generally with the duty and
responsibility of assessing all real property subject to taxation within the
city.{S}
{B}§ 11-202 Maps and records; surveyor.
{S}The commissioner of finance shall appoint a surveyor who shall make the
necessary surveys and corrections of the block or ward maps, and also make all
new tax maps which may be required.{S}
{B}§ 11-203 Maps and records; tax maps.
   a.{S}   As used in the charter and in the code, the term "tax maps" shall mean
and include the block map of taxes and assessments to the extent that the
territory within the city of New York is or shall be embraced in such map, such
ward or land maps as embrace the remainder of such city, and also such maps as
may be prepared under and pursuant to subdivision d of this section.{S}
   b.{S}   Each separately assessed parcel shall be indicated on the tax maps by a
parcel number or by an identification number.{S} A separate identification number
shall be entered upon the tax maps in such manner as clearly to indicate each
separately assessed parcel of real property not indicated by parcel numbering.{S}
Real property indicated by a single identification number shall be deemed to be
a separately assessed parcel.{S}"""


def get_few_shot_example() -> tuple[str, str]:
    """
    Get the few-shot example for NYC tax law chunking.

    Returns:
        Tuple of (input_text, expected_output_with_splits)
    """
    return EXAMPLE_INPUT, EXAMPLE_OUTPUT


def format_user_message(text: str) -> str:
    """
    Format user text for the chunking prompt.

    Args:
        text: Raw text to be chunked.

    Returns:
        Formatted message for the Claude API.
    """
    return f"""Please chunk the following NYC tax law text by inserting split tokens:

{text}"""
