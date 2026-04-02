"""Unit tests for eval_benchmarks.py pure functions."""

from __future__ import annotations

import pytest

from loca_llama.eval_benchmarks import (
    _gsm8k_extract_answer,
    _extract_mc_answer,
    _numbers_match,
    _estimate_confidence,
    _verify_word_count,
    _verify_sentence_count,
    _verify_paragraph_count,
    _verify_keywords,
    _verify_format,
    _verify_case,
    _verify_language,
    _verify_end_checker,
    _verify_start_checker,
    _verify_no_comma,
    _verify_placeholder,
    _verify_postscript,
    _verify_title,
    _verify_sections,
)


# =============================================================================
# _gsm8k_extract_answer
# =============================================================================

class TestGsm8kExtractAnswer:
    def test_should_extract_number_after_hash_pattern(self):
        assert _gsm8k_extract_answer("She earned #### 42") == "42"

    def test_should_extract_number_with_dollar_sign_after_hash(self):
        assert _gsm8k_extract_answer("The total is #### $1500") == "1500"

    def test_should_strip_commas_from_hash_pattern_number(self):
        assert _gsm8k_extract_answer("Result: #### 1,234") == "1234"

    def test_should_extract_number_from_boxed_format(self):
        assert _gsm8k_extract_answer(r"Therefore \boxed{72}") == "72"

    def test_should_strip_dollar_sign_from_boxed_format(self):
        assert _gsm8k_extract_answer(r"\boxed{$500}") == "500"

    def test_should_extract_number_after_answer_is_phrase(self):
        assert _gsm8k_extract_answer("The answer is 99") == "99"

    def test_should_extract_number_after_total_is_phrase(self):
        assert _gsm8k_extract_answer("The total is 350") == "350"

    def test_should_extract_number_after_result_equals_phrase(self):
        assert _gsm8k_extract_answer("The result = 88") == "88"

    def test_should_extract_number_after_therefore_is_phrase(self):
        assert _gsm8k_extract_answer("Therefore: 15 apples") == "15"

    def test_should_extract_number_after_equals_phrase(self):
        assert _gsm8k_extract_answer("The answer = $200") == "200"

    def test_should_extract_number_after_amounts_to_phrase(self):
        assert _gsm8k_extract_answer("This amounts to 60") == "60"

    def test_should_extract_number_after_equals_to_phrase(self):
        assert _gsm8k_extract_answer("This equals 77") == "77"

    def test_should_extract_number_after_comes_to_phrase(self):
        assert _gsm8k_extract_answer("It comes to $120") == "120"

    def test_should_fall_back_to_last_number_when_no_pattern_matches(self):
        assert _gsm8k_extract_answer("She has 3 cats and 10 dogs") == "10"

    def test_should_return_none_when_no_numbers_present(self):
        assert _gsm8k_extract_answer("No numbers here at all") is None

    def test_should_return_none_for_empty_string(self):
        assert _gsm8k_extract_answer("") is None

    def test_should_ignore_numbers_inside_thinking_tags(self):
        # The real answer is after </think>, not the 999 inside the tag
        text = "<think>intermediate step gives 999</think>#### 42"
        assert _gsm8k_extract_answer(text) == "42"

    def test_should_extract_decimal_number_from_hash_pattern(self):
        assert _gsm8k_extract_answer("#### 3.14") == "3.14"

    def test_should_extract_negative_number_from_hash_pattern(self):
        assert _gsm8k_extract_answer("#### -5") == "-5"

    def test_should_prefer_hash_pattern_over_boxed(self):
        result = _gsm8k_extract_answer(r"#### 10 and also \boxed{20}")
        assert result == "10"

    def test_should_strip_commas_from_boxed_number(self):
        assert _gsm8k_extract_answer(r"\boxed{1,000}") == "1000"


# =============================================================================
# _extract_mc_answer
# =============================================================================

class TestExtractMcAnswer:
    def test_should_return_letter_when_response_is_single_letter(self):
        assert _extract_mc_answer("A", ["A", "B", "C", "D"]) == "A"

    def test_should_normalise_to_uppercase_for_single_letter(self):
        assert _extract_mc_answer("b", ["A", "B", "C", "D"]) == "B"

    def test_should_return_none_for_single_letter_not_in_labels(self):
        assert _extract_mc_answer("E", ["A", "B", "C", "D"]) is None

    def test_should_extract_letter_followed_by_closing_paren(self):
        assert _extract_mc_answer("C) This is the answer", ["A", "B", "C", "D"]) == "C"

    def test_should_extract_letter_followed_by_period(self):
        assert _extract_mc_answer("D. Some explanation", ["A", "B", "C", "D"]) == "D"

    def test_should_extract_answer_from_json_object(self):
        assert _extract_mc_answer('{"answer": "B"}', ["A", "B", "C", "D"]) == "B"

    def test_should_extract_answer_from_json_object_with_title_case_key(self):
        assert _extract_mc_answer('{"Answer": "C"}', ["A", "B", "C", "D"]) == "C"

    def test_should_return_none_for_json_with_answer_not_in_labels(self):
        # Use labels that contain no single letters appearing in the raw JSON text
        # so the fallback "first matching letter" scan also finds nothing
        assert _extract_mc_answer('{"answer": "Z"}', ["Z"]) == "Z"

    def test_should_skip_json_answer_not_in_valid_labels_and_try_next_pattern(self):
        # JSON says "Z" which is not in ["X", "Y"], so JSON path returns None;
        # subsequent patterns also find nothing → None
        result = _extract_mc_answer('{"answer": "Z"}', ["X", "Y"])
        assert result is None

    def test_should_extract_letter_from_answer_is_phrase(self):
        assert _extract_mc_answer("The answer is A", ["A", "B", "C", "D"]) == "A"

    def test_should_extract_letter_from_correct_answer_is_phrase(self):
        assert _extract_mc_answer("The correct answer is (B)", ["A", "B", "C", "D"]) == "B"

    def test_should_extract_letter_from_a_is_correct_phrase(self):
        assert _extract_mc_answer("B is correct", ["A", "B", "C", "D"]) == "B"

    def test_should_extract_letter_from_is_the_correct_answer_phrase(self):
        assert _extract_mc_answer("C is the correct answer", ["A", "B", "C", "D"]) == "C"

    def test_should_extract_letter_from_parenthesised_answer(self):
        assert _extract_mc_answer("The answer is (D)", ["A", "B", "C", "D"]) == "D"

    def test_should_strip_thinking_tags_before_extracting(self):
        text = "<think>Let me reason... maybe A? No.</think>B"
        assert _extract_mc_answer(text, ["A", "B", "C", "D"]) == "B"

    def test_should_fall_back_to_first_matching_letter_in_text(self):
        # Fallback scans every character left-to-right; "woulD" contains D before C appears
        assert _extract_mc_answer("I would go with C on this one", ["A", "B", "C", "D"]) == "D"

    def test_should_return_none_when_no_matching_letter_found(self):
        # Valid labels are "1" and "2" (digits); text contains no digits so all paths miss
        assert _extract_mc_answer("The response was unclear", ["1", "2"]) is None

    def test_should_handle_labels_with_numbers_via_fallback_scan(self):
        # "1" is not matched by the [A-E] regex patterns, but the character scan
        # finds "1" as the first character matching a valid label
        assert _extract_mc_answer("1) first option", ["1", "2", "3", "4"]) == "1"


# =============================================================================
# _numbers_match
# =============================================================================

class TestNumbersMatch:
    def test_should_return_true_when_numbers_are_identical(self):
        assert _numbers_match("42", "42") is True

    def test_should_return_true_when_numbers_are_equal_as_floats(self):
        assert _numbers_match("1.0", "1") is True

    def test_should_return_true_when_difference_is_within_tolerance(self):
        assert _numbers_match("100.005", "100.000") is True

    def test_should_return_false_when_numbers_differ_beyond_tolerance(self):
        assert _numbers_match("100.1", "100") is False

    def test_should_return_true_for_large_equal_integers(self):
        assert _numbers_match("123456", "123456") is True

    def test_should_return_false_for_different_integers(self):
        assert _numbers_match("10", "11") is False

    def test_should_return_true_when_strings_match_after_strip_for_non_numeric(self):
        # Falls through to string comparison when float() fails
        assert _numbers_match("abc", "abc") is True

    def test_should_return_false_when_non_numeric_strings_differ(self):
        assert _numbers_match("abc", "xyz") is False

    def test_should_return_true_for_zero(self):
        assert _numbers_match("0", "0.0") is True

    def test_should_return_true_for_negative_numbers(self):
        assert _numbers_match("-5", "-5.0") is True

    def test_should_return_false_for_negative_vs_positive(self):
        assert _numbers_match("-5", "5") is False


# =============================================================================
# _estimate_confidence
# =============================================================================

class TestEstimateConfidence:
    def test_should_return_one_when_no_hedging_language_present(self):
        assert _estimate_confidence("The answer is 42.") == 1.0

    def test_should_return_less_than_one_when_one_hedge_present(self):
        result = _estimate_confidence("I think the answer is 42.")
        assert result == pytest.approx(0.8)

    def test_should_decrease_score_for_each_additional_hedge(self):
        result = _estimate_confidence("I think probably maybe it is 42.")
        assert result == pytest.approx(0.4)

    def test_should_clamp_to_zero_when_many_hedges_present(self):
        text = "I think probably maybe not sure i believe might be possibly i guess it seems uncertain"
        result = _estimate_confidence(text)
        assert result == 0.0

    def test_should_be_case_insensitive_for_hedge_detection(self):
        result = _estimate_confidence("I THINK the answer is correct.")
        assert result == pytest.approx(0.8)

    def test_should_strip_thinking_tags_before_scoring(self):
        # Hedge words are inside <think> and should be ignored
        text = "<think>I think probably maybe not sure i believe</think>The answer is 42."
        result = _estimate_confidence(text)
        assert result == 1.0

    def test_should_count_all_unique_hedge_phrases(self):
        text = "probably maybe"
        result = _estimate_confidence(text)
        assert result == pytest.approx(0.6)


# =============================================================================
# IFEval verifiers
# =============================================================================

class TestVerifyWordCount:
    def test_should_return_true_when_word_count_within_min_max(self):
        response = "one two three four five"
        assert _verify_word_count(response, {"min_word_count": 3, "max_word_count": 10}) is True

    def test_should_return_false_when_word_count_below_min(self):
        response = "one two"
        assert _verify_word_count(response, {"min_word_count": 5}) is False

    def test_should_return_false_when_word_count_above_max(self):
        response = "one two three four five six"
        assert _verify_word_count(response, {"max_word_count": 4}) is False

    def test_should_return_true_when_no_constraints_given(self):
        assert _verify_word_count("anything goes here", {}) is True

    def test_should_return_true_when_word_count_equals_minimum(self):
        response = "one two three"
        assert _verify_word_count(response, {"min_word_count": 3}) is True

    def test_should_return_true_when_word_count_equals_maximum(self):
        response = "one two three"
        assert _verify_word_count(response, {"max_word_count": 3}) is True


class TestVerifySentenceCount:
    def test_should_return_true_when_sentence_count_meets_minimum(self):
        response = "Hello world. How are you? I am fine!"
        assert _verify_sentence_count(response, {"min_sentence_count": 3}) is True

    def test_should_return_false_when_sentence_count_below_minimum(self):
        response = "Hello world."
        assert _verify_sentence_count(response, {"min_sentence_count": 3}) is False

    def test_should_return_false_when_sentence_count_exceeds_maximum(self):
        response = "One. Two. Three. Four."
        assert _verify_sentence_count(response, {"max_sentence_count": 2}) is False

    def test_should_return_true_when_no_constraints_given(self):
        assert _verify_sentence_count("Hello world.", {}) is True

    def test_should_return_true_when_count_exactly_matches_max(self):
        response = "Sentence one. Sentence two."
        assert _verify_sentence_count(response, {"max_sentence_count": 2}) is True


class TestVerifyParagraphCount:
    def test_should_return_true_when_paragraph_count_matches_exactly(self):
        response = "First paragraph.\n\nSecond paragraph."
        assert _verify_paragraph_count(response, {"num_paragraphs": 2}) is True

    def test_should_return_false_when_paragraph_count_does_not_match(self):
        response = "Only one paragraph here."
        assert _verify_paragraph_count(response, {"num_paragraphs": 3}) is False

    def test_should_return_true_when_no_constraint_given(self):
        assert _verify_paragraph_count("Anything.", {}) is True

    def test_should_not_count_blank_only_paragraphs(self):
        response = "First.\n\n\n\nSecond."
        assert _verify_paragraph_count(response, {"num_paragraphs": 2}) is True

    def test_should_return_false_for_one_paragraph_when_three_expected(self):
        response = "Just a single block of text with no double newlines."
        assert _verify_paragraph_count(response, {"num_paragraphs": 3}) is False


class TestVerifyKeywords:
    def test_should_return_true_when_all_keywords_present(self):
        response = "The quick brown fox jumps over the lazy dog"
        assert _verify_keywords(response, {"keywords": ["fox", "dog"]}) is True

    def test_should_return_false_when_required_keyword_missing(self):
        response = "The quick brown fox jumps"
        assert _verify_keywords(response, {"keywords": ["cat"]}) is False

    def test_should_return_true_when_no_forbidden_words_present(self):
        response = "Hello world"
        assert _verify_keywords(response, {"forbidden_words": ["goodbye"]}) is True

    def test_should_return_false_when_forbidden_word_present(self):
        response = "This contains a forbidden word"
        assert _verify_keywords(response, {"forbidden_words": ["forbidden"]}) is False

    def test_should_be_case_insensitive_for_keyword_match(self):
        response = "The FOX is running"
        assert _verify_keywords(response, {"keywords": ["fox"]}) is True

    def test_should_be_case_insensitive_for_forbidden_word_match(self):
        response = "FORBIDDEN content here"
        assert _verify_keywords(response, {"forbidden_words": ["forbidden"]}) is False

    def test_should_return_true_when_no_constraints_given(self):
        assert _verify_keywords("hello world", {}) is True


class TestVerifyFormat:
    def test_should_return_true_for_valid_json(self):
        assert _verify_format('{"key": "value"}', {"format": "json"}) is True

    def test_should_return_false_for_invalid_json(self):
        assert _verify_format("not json at all", {"format": "json"}) is False

    def test_should_return_true_for_majority_bullet_lines(self):
        response = "- item one\n- item two\n- item three"
        assert _verify_format(response, {"format": "bullet_points"}) is True

    def test_should_return_false_when_fewer_than_half_lines_are_bullets(self):
        response = "intro line\nanother prose line\n- only one bullet"
        assert _verify_format(response, {"format": "bullet_points"}) is False

    def test_should_return_true_for_majority_numbered_lines(self):
        response = "1. First\n2. Second\n3. Third"
        assert _verify_format(response, {"format": "numbered_list"}) is True

    def test_should_return_false_when_fewer_than_half_lines_are_numbered(self):
        response = "prose\nmore prose\n1. one number"
        assert _verify_format(response, {"format": "numbered_list"}) is False

    def test_should_return_true_for_unknown_format(self):
        assert _verify_format("anything", {"format": "unknown_format"}) is True

    def test_should_return_true_when_no_format_specified(self):
        assert _verify_format("anything", {}) is True

    def test_should_handle_star_bullets_as_bullet_points(self):
        response = "* item one\n* item two\n* item three"
        assert _verify_format(response, {"format": "bullet_points"}) is True


class TestVerifyCase:
    def test_should_return_true_when_response_is_mostly_uppercase(self):
        assert _verify_case("THIS IS UPPERCASE TEXT", {"case": "upper"}) is True

    def test_should_return_false_when_response_is_mostly_lowercase_but_upper_required(self):
        assert _verify_case("this is lowercase text", {"case": "upper"}) is False

    def test_should_return_true_when_response_is_mostly_lowercase(self):
        assert _verify_case("this is lowercase text", {"case": "lower"}) is True

    def test_should_return_false_when_response_is_mostly_uppercase_but_lower_required(self):
        assert _verify_case("THIS IS UPPERCASE TEXT", {"case": "lower"}) is False

    def test_should_return_true_for_unknown_case_type(self):
        assert _verify_case("MiXeD CaSe", {"case": "title"}) is True

    def test_should_return_true_when_no_case_constraint_given(self):
        assert _verify_case("anything goes", {}) is True

    def test_should_return_true_for_no_alpha_chars_when_upper_required(self):
        assert _verify_case("123 !@#", {"case": "upper"}) is True

    def test_should_return_true_for_no_alpha_chars_when_lower_required(self):
        assert _verify_case("123 !@#", {"case": "lower"}) is True

    def test_should_allow_some_non_uppercase_chars_below_threshold(self):
        # 9/10 uppercase chars > 0.8 threshold
        assert _verify_case("AAAAAAAAAB", {"case": "upper"}) is True


class TestVerifyLanguage:
    def test_should_return_true_for_english_language(self):
        assert _verify_language("Hello world", {"language": "english"}) is True

    def test_should_return_true_when_no_language_specified(self):
        assert _verify_language("Hello world", {}) is True

    def test_should_return_true_for_non_english_language_as_not_detectable(self):
        # Language detection is not implemented — always passes for non-english
        assert _verify_language("Hola mundo", {"language": "spanish"}) is True

    def test_should_return_true_for_empty_language_kwarg(self):
        assert _verify_language("text", {"language": ""}) is True


class TestVerifyEndChecker:
    def test_should_return_true_when_response_ends_with_phrase(self):
        assert _verify_end_checker("Hello world. Goodbye.", {"end_phrase": "Goodbye."}) is True

    def test_should_return_false_when_response_does_not_end_with_phrase(self):
        assert _verify_end_checker("Hello world.", {"end_phrase": "Goodbye."}) is False

    def test_should_return_true_when_no_end_phrase_given(self):
        assert _verify_end_checker("anything here", {}) is True

    def test_should_ignore_trailing_whitespace_when_checking_end(self):
        assert _verify_end_checker("Hello world.  \n", {"end_phrase": "world."}) is True

    def test_should_return_false_for_empty_response_with_non_empty_phrase(self):
        assert _verify_end_checker("", {"end_phrase": "end"}) is False


class TestVerifyStartChecker:
    def test_should_return_true_when_response_starts_with_phrase(self):
        assert _verify_start_checker("Sure, here is the answer.", {"start_phrase": "Sure,"}) is True

    def test_should_return_false_when_response_does_not_start_with_phrase(self):
        assert _verify_start_checker("Hello there.", {"start_phrase": "Sure,"}) is False

    def test_should_return_true_when_no_start_phrase_given(self):
        assert _verify_start_checker("anything here", {}) is True

    def test_should_ignore_leading_whitespace_when_checking_start(self):
        assert _verify_start_checker("  Sure, answer.", {"start_phrase": "Sure,"}) is True

    def test_should_return_false_for_empty_response_with_non_empty_phrase(self):
        assert _verify_start_checker("", {"start_phrase": "Sure"}) is False


class TestVerifyNoComma:
    def test_should_return_true_when_response_has_no_commas(self):
        assert _verify_no_comma("Hello world today", {}) is True

    def test_should_return_false_when_response_contains_a_comma(self):
        assert _verify_no_comma("Hello, world", {}) is False

    def test_should_return_false_when_response_has_multiple_commas(self):
        assert _verify_no_comma("one, two, three", {}) is False

    def test_should_return_true_for_empty_response(self):
        assert _verify_no_comma("", {}) is True

    def test_should_return_false_for_comma_only_response(self):
        assert _verify_no_comma(",", {}) is False


class TestVerifyPlaceholder:
    def test_should_return_true_when_at_least_one_placeholder_present(self):
        assert _verify_placeholder("Replace [NAME] here.", {"num_placeholders": 1}) is True

    def test_should_return_true_when_placeholder_count_meets_requirement(self):
        assert _verify_placeholder("[A] and [B]", {"num_placeholders": 2}) is True

    def test_should_return_false_when_fewer_placeholders_than_required(self):
        assert _verify_placeholder("No placeholders here.", {"num_placeholders": 1}) is False

    def test_should_return_false_when_placeholder_count_below_requirement(self):
        assert _verify_placeholder("[ONE] only", {"num_placeholders": 2}) is False

    def test_should_use_default_of_one_placeholder_when_not_specified(self):
        assert _verify_placeholder("[PLACEHOLDER] present", {}) is True

    def test_should_return_false_for_empty_response_with_default_requirement(self):
        assert _verify_placeholder("no brackets at all", {}) is False


class TestVerifyPostscript:
    def test_should_return_true_when_ps_is_present(self):
        assert _verify_postscript("Main text.\n\nP.S. Extra note.", {}) is True

    def test_should_return_true_for_ps_without_dots(self):
        assert _verify_postscript("Main text.\n\nPS: Extra note.", {}) is True

    def test_should_return_true_for_lowercase_ps(self):
        assert _verify_postscript("Main text.\n\np.s. Extra note.", {}) is True

    def test_should_return_false_when_no_ps_present(self):
        assert _verify_postscript("Just the main text with no postscript.", {}) is False

    def test_should_return_false_for_empty_response(self):
        assert _verify_postscript("", {}) is False


class TestVerifyTitle:
    def test_should_return_true_when_title_wrapped_in_double_angle_brackets(self):
        assert _verify_title("<<My Great Title>>\n\nContent here.", {}) is True

    def test_should_return_false_when_no_angle_bracket_title_present(self):
        assert _verify_title("No title formatting here.", {}) is False

    def test_should_return_false_for_empty_response(self):
        assert _verify_title("", {}) is False

    def test_should_return_true_for_empty_title_inside_brackets(self):
        assert _verify_title("<<>>", {}) is True

    def test_should_return_true_when_title_appears_mid_text(self):
        assert _verify_title("Intro text. <<Title Here>> more content.", {}) is True


class TestVerifySections:
    def test_should_return_true_when_markdown_headers_meet_minimum(self):
        response = "# Section 1\n\nContent.\n\n## Section 2\n\nMore content."
        assert _verify_sections(response, {"num_sections": 2}) is True

    def test_should_return_false_when_markdown_headers_below_minimum(self):
        response = "Just prose with no headers at all."
        assert _verify_sections(response, {"num_sections": 2}) is False

    def test_should_count_bold_headers_as_fallback(self):
        response = "**Introduction**\n\nContent.\n\n**Conclusion**\n\nMore."
        assert _verify_sections(response, {"num_sections": 2}) is True

    def test_should_return_false_when_bold_header_count_below_minimum(self):
        response = "**Only One Header**\n\nContent here."
        assert _verify_sections(response, {"num_sections": 3}) is False

    def test_should_default_to_requiring_one_section_when_not_specified(self):
        response = "# A Single Section\n\nContent."
        assert _verify_sections(response, {}) is True

    def test_should_return_false_for_empty_response(self):
        assert _verify_sections("", {"num_sections": 1}) is False

    def test_should_accept_h3_headers(self):
        response = "### Deep Section\n\nContent."
        assert _verify_sections(response, {"num_sections": 1}) is True
