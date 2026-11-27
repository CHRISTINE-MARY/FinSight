from datetime import date
from uuid import UUID

import pytest

from app.assistant.outputs import Citation, GroundedAnswer
from app.grounding.tables import year_column_lines
from app.grounding.validator import GroundingError, validate
from app.retrieval.retriever import NeighborChunk, RetrievedPassage

CHUNK_ID = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
OTHER_ID = UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
CONTENT = "iPhone net sales were $201,183 million."


def test_citation_must_come_from_a_retrieved_passage() -> None:
    answer = GroundedAnswer(
        answer="iPhone sales were $201,183 million.",
        citations=[Citation(chunk_id=OTHER_ID, excerpt="iPhone net sales")],
    )

    with pytest.raises(GroundingError, match="not retrieved"):
        validate(answer, {CHUNK_ID: _passage()})


def test_factual_answer_without_a_citation_is_rejected() -> None:
    answer = GroundedAnswer(answer="iPhone sales grew.")

    with pytest.raises(GroundingError, match="no citation"):
        validate(answer, {CHUNK_ID: _passage()})


def test_not_enough_evidence_may_have_no_citations() -> None:
    answer = GroundedAnswer(answer="The corpus does not contain enough evidence.")

    validated = validate(answer, {})

    assert validated.citations == ()


def test_valid_citation_uses_the_retrieved_filing_not_the_model() -> None:
    answer = GroundedAnswer(
        answer="iPhone net sales were $201,183 million.",
        citations=[
            Citation(
                chunk_id=CHUNK_ID, excerpt="iPhone net sales were $201,183 million."
            )
        ],
    )

    validated = validate(answer, {CHUNK_ID: _passage()})

    assert validated.citations[0].chunk_id == CHUNK_ID
    assert validated.citations[0].data["ticker"] == "AAPL"
    assert validated.citations[0].data["filingType"] == "10-K"
    assert validated.citations[0].data["page"] == "42"
    assert validated.citations[0].data["companyName"] == "Apple Inc."


def test_excerpt_missing_from_the_passage_is_rejected() -> None:
    answer = GroundedAnswer(
        answer="Services grew.",
        citations=[Citation(chunk_id=CHUNK_ID, excerpt="Services grew 14 percent")],
    )

    with pytest.raises(GroundingError, match="does not support"):
        validate(answer, {CHUNK_ID: _passage()})


TABLE = (
    "| Mac                             | Mac                             | Mac"
    "                             | 29,984 |  29,984 | (27)   | (27)   | %      |"
)
IPAD = "| iPad                            | iPad                            | iPad                            | 26,694"
SECOND_ID = UUID("dddddddd-dddd-4ddd-8ddd-dddddddddddd")


def test_backend_copies_the_stored_chunk() -> None:
    answer = GroundedAnswer(
        answer="Mac net sales were 29,984.",
        citations=[Citation(chunk_id=CHUNK_ID, excerpt="")],
    )

    validated = validate(answer, {CHUNK_ID: _passage(TABLE)})

    assert validated.citations[0].excerpt == TABLE
    assert validated.citations[0].data["excerpt"] == TABLE
    assert validated.citations[0].data["content"] == TABLE


def test_corrupted_model_excerpt_is_replaced_by_the_chunk() -> None:
    answer = GroundedAnswer(
        answer="Mac net sales were 29,984.",
        citations=[
            Citation(
                chunk_id=CHUNK_ID,
                excerpt="Mac Mac Mac 29,984 | (27) | (29,357",
            )
        ],
    )

    validated = validate(answer, {CHUNK_ID: _passage(TABLE)})

    assert validated.citations[0].excerpt == TABLE


def test_chunk_id_from_outside_this_turn_is_rejected() -> None:
    answer = GroundedAnswer(
        answer="Mac net sales were 29,984.",
        citations=[Citation(chunk_id=OTHER_ID, excerpt=TABLE)],
    )

    with pytest.raises(GroundingError, match="not retrieved"):
        validate(answer, {CHUNK_ID: _passage(TABLE)})


def test_each_citation_keeps_its_own_chunk() -> None:
    answer = GroundedAnswer(
        answer="Mac was 29,984 and iPad was 26,694.",
        citations=[
            Citation(chunk_id=CHUNK_ID, excerpt="wrong mac row"),
            Citation(chunk_id=SECOND_ID, excerpt="wrong ipad row"),
        ],
    )
    retrieved = {
        CHUNK_ID: _passage(TABLE),
        SECOND_ID: _passage(IPAD, SECOND_ID),
    }

    validated = validate(answer, retrieved)

    assert validated.citations[0].chunk_id == CHUNK_ID
    assert validated.citations[0].excerpt == TABLE
    assert validated.citations[1].chunk_id == SECOND_ID
    assert validated.citations[1].excerpt == IPAD
    assert "$" not in IPAD
    assert validated.citations[1].excerpt == IPAD


def test_table_excerpt_keeps_pipes_dollars_and_empty_cells() -> None:
    services = (
        "| Services (1)                    | Services (1)"
        "                    | 96,169 |  96,169 |      |    |    |    | 13     | 13     | %      |"
    )
    answer = GroundedAnswer(
        answer="Services net sales were 96,169.",
        citations=[
            Citation(
                chunk_id=CHUNK_ID,
                excerpt="|        |        |        |",
            )
        ],
    )

    validated = validate(answer, {CHUNK_ID: _passage(services)})

    assert validated.citations[0].excerpt == services
    assert "|        |" not in validated.citations[0].excerpt
    assert "(1)" in validated.citations[0].excerpt
    assert "13     | 13     | %      |" in validated.citations[0].excerpt


def test_chunk_id_alone_does_not_support_an_unrelated_answer() -> None:
    answer = GroundedAnswer(
        answer="Services grew 14 percent.",
        citations=[Citation(chunk_id=CHUNK_ID, excerpt="")],
    )

    with pytest.raises(GroundingError, match="does not support"):
        validate(answer, {CHUNK_ID: _passage()})


HEADER = "|  |  | 2024 | 2024 | Change | 2023 | 2023 |"
MAC_LINE = "| Mac | Mac | 29,984 |  29,984 | 2 | 29,357 |  29,357 |"
IPAD_LINE = "| iPad | iPad | 26,694 |  26,694 | (6) | 28,300 |  28,300 |"
IPHONE_LINE = "| iPhone | iPhone | $ | 201,183 | (2) | $ | 200,583 |"
ROWS = f"{MAC_LINE}\n{IPAD_LINE}"
HEADER_ID = UUID("eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee")
DOC_ID = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
QUESTION = "What were Apple's iPad net sales for fiscal 2024?"


def test_locator_slices_the_original_line() -> None:
    answer = GroundedAnswer(
        answer="iPad net sales were $26,694 million.",
        citations=[
            Citation(
                chunk_id=CHUNK_ID,
                label="iPad",
                figure="26,694",
                excerpt="| iPad | iPad | $ | 26,694 |",
            )
        ],
    )

    validated = validate(
        answer, {CHUNK_ID: _passage(ROWS)}, QUESTION
    )

    assert validated.citations[0].excerpt == IPAD_LINE
    assert validated.citations[0].data["content"] == ROWS
    assert "$" not in validated.citations[0].excerpt
    assert "|  26,694 |" in validated.citations[0].excerpt


def test_locator_from_another_chunk_is_rejected() -> None:
    answer = GroundedAnswer(
        answer="iPad net sales were $26,694 million.",
        citations=[Citation(chunk_id=CHUNK_ID, label="iPad", figure="26,694")],
    )
    retrieved = {
        CHUNK_ID: _passage(MAC_LINE),
        SECOND_ID: _passage(IPAD_LINE, SECOND_ID),
    }

    with pytest.raises(GroundingError, match="Citation mistake"):
        validate(answer, retrieved, QUESTION)


def test_a_locator_in_no_retrieved_chunk_is_a_hallucination() -> None:
    answer = GroundedAnswer(
        answer="The offices are at One Apple Park Way.",
        citations=[
            Citation(chunk_id=CHUNK_ID, label="One Apple Park Way", figure="")
        ],
    )

    with pytest.raises(GroundingError, match="Hallucination"):
        validate(answer, {CHUNK_ID: _passage("The headquarters is in Cupertino.")})


def test_wrong_year_column_is_rejected() -> None:
    answer = GroundedAnswer(
        answer="Apple reported iPad net sales of $28,300 million for fiscal 2024.",
        citations=[
            Citation(chunk_id=CHUNK_ID, label="iPad", figure="28,300", excerpt="28,300")
        ],
    )

    with pytest.raises(GroundingError, match="2023 column"):
        validate(answer, _row_with_header(), QUESTION)


def test_year_header_in_the_previous_chunk_accepts_the_2024_cell() -> None:
    answer = GroundedAnswer(
        answer="Apple reported iPad net sales of $26,694 million for fiscal 2024.",
        citations=[Citation(chunk_id=CHUNK_ID, label="iPad", figure="$26,694")],
    )

    validated = validate(answer, _row_with_header(), QUESTION)

    assert validated.citations[0].excerpt == IPAD_LINE
    assert "28,300" in validated.citations[0].excerpt
    assert validated.citations[0].excerpt.startswith("| iPad")


def test_ranked_header_chunk_still_maps_the_cited_row() -> None:
    row = _passage(ROWS, chunk_index=81)
    header = _passage(HEADER, HEADER_ID, chunk_index=80)
    answer = GroundedAnswer(
        answer="Apple reported iPad net sales of $26,694 million for fiscal 2024.",
        citations=[Citation(chunk_id=CHUNK_ID, label="iPad", figure="26,694")],
    )

    validated = validate(answer, {CHUNK_ID: row, HEADER_ID: header}, "FY2024 iPad net sales")

    assert validated.citations[0].excerpt == IPAD_LINE


def test_dollar_cell_stays_in_its_own_column() -> None:
    rows = f"{HEADER}\n{IPHONE_LINE}"
    wrong = GroundedAnswer(
        answer="iPhone net sales were $200,583 million in fiscal 2024.",
        citations=[Citation(chunk_id=CHUNK_ID, label="iPhone", figure="200,583")],
    )
    right = GroundedAnswer(
        answer="iPhone net sales were $201,183 million in fiscal 2024.",
        citations=[Citation(chunk_id=CHUNK_ID, label="iPhone", figure="201,183")],
    )

    with pytest.raises(GroundingError, match="2023 column"):
        validate(wrong, {CHUNK_ID: _passage(rows)}, "What were iPhone net sales for fiscal 2024?")
    validated = validate(
        right, {CHUNK_ID: _passage(rows)}, "What were iPhone net sales for fiscal 2024?"
    )

    assert validated.citations[0].excerpt == IPHONE_LINE
    assert "| $ | 201,183 |" in validated.citations[0].excerpt


def test_each_locator_keeps_its_own_line() -> None:
    answer = GroundedAnswer(
        answer="Mac was 29,984 and iPad was 26,694.",
        citations=[
            Citation(chunk_id=CHUNK_ID, label="Mac", figure="29,984", excerpt=IPAD_LINE),
            Citation(
                chunk_id=SECOND_ID, label="iPad", figure="26,694", excerpt=MAC_LINE
            ),
        ],
    )
    retrieved = {
        CHUNK_ID: _passage(f"{HEADER}\n{MAC_LINE}"),
        SECOND_ID: _passage(f"{HEADER}\n{IPAD_LINE}", SECOND_ID),
    }

    validated = validate(answer, retrieved, "Show Mac and iPad for fiscal 2024")

    assert validated.citations[0].excerpt == MAC_LINE
    assert validated.citations[1].excerpt == IPAD_LINE
    assert "iPad" not in validated.citations[0].excerpt
    assert "Mac" not in validated.citations[1].excerpt


def test_empty_cells_and_pipes_are_not_rewritten() -> None:
    services = (
        "| Services (1)                    | Services (1)"
        "                    | 96,169 |  96,169 |      |    |    |    | 13     | 13     | %      |"
    )
    answer = GroundedAnswer(
        answer="Services net sales were 96,169.",
        citations=[
            Citation(
                chunk_id=CHUNK_ID,
                label="Services",
                figure="96,169",
                excerpt="|        |        |",
            )
        ],
    )

    validated = validate(answer, {CHUNK_ID: _passage(services)})

    assert validated.citations[0].excerpt == services
    assert "13     | 13     | %      |" in validated.citations[0].excerpt


FISCAL_YEAR = (
    "The Company's fiscal year is the 52- or 53-week period that ends on the "
    "last Saturday of September."
)
GPU = "Our invention of the GPU in 1999 sparked the growth of the PC gaming market."
RESEARCH = (
    "We have invested over $58.2 billion in research and development since our inception."
)
OUTSIDE_US = (
    "Revenue from sales to customers outside of the United States accounted for "
    "53% and 56% of total revenue for fiscal years 2025 and 2024, respectively."
)


def test_sentence_figure_with_extra_words_matches_the_number() -> None:
    answer = GroundedAnswer(
        answer="NVIDIA has invested over $58.2 billion in research and development.",
        citations=[Citation(chunk_id=CHUNK_ID, figure="$58.2 billion")],
    )

    validated = validate(answer, {CHUNK_ID: _passage(RESEARCH)})

    assert validated.citations[0].excerpt == RESEARCH


def test_plain_year_and_percent_match_a_sentence() -> None:
    invented = GroundedAnswer(
        answer="NVIDIA invented the GPU in 1999.",
        citations=[Citation(chunk_id=CHUNK_ID, figure="1999")],
    )
    share = GroundedAnswer(
        answer="53% of NVIDIA revenue came from outside the United States in fiscal 2025.",
        citations=[Citation(chunk_id=CHUNK_ID, figure="53%")],
    )

    assert validate(invented, {CHUNK_ID: _passage(GPU)}).citations[0].excerpt == GPU
    validated = validate(
        share,
        {CHUNK_ID: _passage(OUTSIDE_US)},
        "What share of NVIDIA revenue came from outside the United States?",
    )
    assert "53%" in validated.citations[0].excerpt


def test_non_numeric_figure_keeps_the_date_sentence() -> None:
    answer = GroundedAnswer(
        answer="Apple's fiscal year ends on the last Saturday of September.",
        citations=[Citation(chunk_id=CHUNK_ID, label="fiscal year", figure="September")],
    )

    validated = validate(answer, {CHUNK_ID: _passage(FISCAL_YEAR)}, "When does Apple's fiscal year end?")

    assert validated.citations[0].excerpt == FISCAL_YEAR


def test_percent_assigned_to_the_other_year_is_rejected() -> None:
    answer = GroundedAnswer(
        answer="56% of NVIDIA revenue came from outside the United States in fiscal 2025.",
        citations=[Citation(chunk_id=CHUNK_ID, figure="56%")],
    )

    with pytest.raises(GroundingError, match="2025"):
        validate(answer, {CHUNK_ID: _passage(OUTSIDE_US)})


def test_swapped_year_columns_are_rejected() -> None:
    answer = GroundedAnswer(
        answer="iPad net sales were 28,300 in 2024 and 26,694 in 2023.",
        citations=[Citation(chunk_id=CHUNK_ID, label="iPad", figure="28,300")],
    )

    with pytest.raises(GroundingError, match="2023 column"):
        validate(
            answer,
            _row_with_header(),
            "Compare Apple iPad net sales for 2024 and 2023.",
        )


def test_date_on_another_line_of_the_same_chunk_is_kept() -> None:
    content = (
        "As of September 28, 2024.\n"
        "The Company had approximately 164,000 full-time equivalent employees."
    )
    answer = GroundedAnswer(
        answer="Apple had approximately 164,000 employees as of September 28, 2024.",
        citations=[Citation(chunk_id=CHUNK_ID, figure="164,000")],
    )

    validated = validate(answer, {CHUNK_ID: _passage(content)})

    assert validated.citations[0].excerpt == (
        "The Company had approximately 164,000 full-time equivalent employees."
    )


def test_split_percent_cell_counts_as_the_percent() -> None:
    row = "| Mac | Mac | 29,984 | 2 | % |"
    answer = GroundedAnswer(
        answer="Mac net sales increased 2% to 29,984.",
        citations=[Citation(chunk_id=CHUNK_ID, label="Mac", figure="29,984")],
    )

    validated = validate(answer, {CHUNK_ID: _passage(row)})

    assert validated.citations[0].excerpt == row


def test_ordinary_fact_does_not_need_a_number() -> None:
    sentence = (
        "We report Google in two segments, Google Services and Google Cloud, "
        "and all non-Google businesses collectively as Other Bets."
    )
    answer = GroundedAnswer(
        answer="Alphabet reports Google Services, Google Cloud, and Other Bets.",
        citations=[Citation(chunk_id=CHUNK_ID, figure="")],
    )

    validated = validate(answer, {CHUNK_ID: _passage(sentence)})

    assert validated.citations[0].excerpt == sentence


def test_year_header_two_chunks_away_still_rejects_the_other_column() -> None:
    wear = "| Wearables | Wearables | 37,005 |  37,005 | (7) | 39,845 |  39,845 |"
    row = _passage(wear, chunk_index=82)
    header = _passage(HEADER, HEADER_ID, chunk_index=80)
    retrieved = {CHUNK_ID: row, HEADER_ID: header}
    right = GroundedAnswer(
        answer="Wearables net sales were 37,005.",
        citations=[Citation(chunk_id=CHUNK_ID, label="Wearables", figure="37,005")],
    )
    wrong = GroundedAnswer(
        answer="Wearables net sales were 39,845.",
        citations=[Citation(chunk_id=CHUNK_ID, label="Wearables", figure="39,845")],
    )

    validated = validate(
        right, retrieved, "What were Apple's Wearables net sales for fiscal 2024?"
    )
    assert validated.citations[0].excerpt == wear

    with pytest.raises(GroundingError, match="2023 column"):
        validate(
            wrong, retrieved, "What were Apple's Wearables net sales for fiscal 2024?"
        )


def test_year_labels_read_a_header_line_inside_a_neighbor_chunk() -> None:
    previous = (
        "net sales by category for 2024, 2023 and 2022 (dollars in millions):\n"
        f"{HEADER}\n"
        f"{IPAD_LINE}"
    )
    wear = "| Wearables | Wearables | 37,005 |  37,005 | (7) | 39,845 |  39,845 |"

    labels = year_column_lines(wear, [previous])

    assert "Wearables 2024: 37,005" in labels
    assert "Wearables 2023: 39,845" in labels


def test_rounded_billion_that_is_not_printed_is_rejected() -> None:
    answer = GroundedAnswer(
        answer=(
            "Apple's iPhone net sales in fiscal 2023 were $200.6 billion, "
            "a 2% decrease from $200.6 billion in fiscal 2022."
        ),
        citations=[Citation(chunk_id=CHUNK_ID, label="iPhone", figure="200,583")],
    )

    with pytest.raises(GroundingError, match="200.6"):
        validate(
            answer,
            {CHUNK_ID: _passage(f"{HEADER}\n{IPHONE_LINE}")},
            "What were Apple's iPhone net sales for fiscal 2023?",
        )


def test_date_copied_from_the_question_need_not_be_in_the_line() -> None:
    content = "The Company had approximately 164,000 full-time equivalent employees."
    answer = GroundedAnswer(
        answer="Apple had approximately 164,000 employees as of September 28, 2024.",
        citations=[Citation(chunk_id=CHUNK_ID, figure="164,000")],
    )

    validated = validate(
        answer,
        {CHUNK_ID: _passage(content)},
        "How many employees did Apple have as of September 28, 2024?",
    )

    assert "164,000" in validated.citations[0].excerpt


def test_date_the_model_adds_must_be_in_the_chunk() -> None:
    content = "The Company had approximately 164,000 full-time equivalent employees."
    answer = GroundedAnswer(
        answer="Apple had 164,000 employees as of the last Saturday of September.",
        citations=[Citation(chunk_id=CHUNK_ID, figure="164,000")],
    )

    with pytest.raises(GroundingError, match="saturday"):
        validate(answer, {CHUNK_ID: _passage(content)}, "How many employees did Apple have?")


def test_repeated_prose_label_keeps_the_whole_chunk() -> None:
    content = (
        "The fiscal year ends on the last Saturday of September.\n"
        "The fiscal year is the 52- or 53-week period."
    )
    answer = GroundedAnswer(
        answer="The fiscal year ends on the last Saturday of September.",
        citations=[Citation(chunk_id=CHUNK_ID, label="fiscal year")],
    )

    validated = validate(
        answer, {CHUNK_ID: _passage(content)}, "When does Apple's fiscal year end?"
    )

    assert validated.citations[0].excerpt == content


def test_comparison_can_cite_both_years_on_the_row() -> None:
    answer = GroundedAnswer(
        answer="iPad net sales were 26,694 in 2024 and 28,300 in 2023.",
        citations=[Citation(chunk_id=CHUNK_ID, label="iPad", figure="26,694")],
    )

    validated = validate(
        answer,
        _row_with_header(),
        "Compare Apple iPad net sales for 2024 and 2023.",
    )

    assert validated.citations[0].excerpt == IPAD_LINE


def _row_with_header() -> dict[UUID, RetrievedPassage]:
    row = _passage(
        ROWS,
        chunk_index=81,
        neighbors=(NeighborChunk(HEADER_ID, 80, HEADER),),
    )
    return {CHUNK_ID: row}


def _passage(
    content: str = CONTENT,
    chunk_id: UUID = CHUNK_ID,
    chunk_index: int = 4,
    neighbors: tuple[NeighborChunk, ...] = (),
) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id,
        document_id=DOC_ID,
        chunk_index=chunk_index,
        content=content,
        token_count=12,
        metadata={"page": 42, "section": ["Net sales"]},
        ticker="AAPL",
        company_name="Apple Inc.",
        filing_type="10-K",
        filing_date=date(2024, 11, 1),
        fiscal_year=2024,
        accession_number="0000320193-24-000123",
        rrf_score=0.03,
        semantic_rank=1,
        lexical_rank=1,
        neighbors=neighbors,
    )
