from triage.redact import ComprehendMedicalRedactor, RegexRedactor, Span, apply_spans

from .conftest import FakeComprehendMedical

TEXT = "This is Dana Whitfield, age 72, call 555-201-7788 or dana@example.com about 03/02/2026."


def test_comprehend_medical_replaces_with_type_tags():
    cm = FakeComprehendMedical({"Dana Whitfield": "NAME", "72": "AGE", "555-201-7788": "PHONE_OR_FAX"})
    red = ComprehendMedicalRedactor(cm).redact(TEXT)
    assert "Dana" not in red.text and "555-201" not in red.text
    assert "This is [NAME], age [AGE], call [PHONE_OR_FAX]" in red.text
    assert red.entity_counts == {"NAME": 1, "AGE": 1, "PHONE_OR_FAX": 1}


def test_low_score_entities_are_ignored():
    class Low(FakeComprehendMedical):
        def detect_phi(self, Text):  # noqa: N803
            return {"Entities": [{"BeginOffset": 0, "EndOffset": 4, "Type": "NAME", "Score": 0.1}]}

    assert ComprehendMedicalRedactor(Low({})).redact(TEXT).text == TEXT


def test_long_text_is_windowed_with_correct_offsets():
    text = ("filler words here. " * 1500) + "Contact Dana Whitfield today."
    cm = FakeComprehendMedical({"Dana Whitfield": "NAME"})
    red = ComprehendMedicalRedactor(cm).redact(text)
    assert cm.calls > 1
    assert red.text.endswith("Contact [NAME] today.")


def test_overlapping_spans():
    red = apply_spans("abcdefgh", [Span(1, 5, "A"), Span(3, 7, "B"), Span(7, 8, "C")], "t")
    assert red.text == "a[A]fg[C]"  # the overlapping span B is dropped
    assert red.entity_counts == {"A": 1, "C": 1}


def test_regex_redactor_catches_contact_details():
    red = RegexRedactor().redact(TEXT + " Dr. Priya Raman, MRN: 00123456")
    for leaked in ("555-201-7788", "dana@example.com", "03/02/2026", "Priya Raman", "00123456"):
        assert leaked not in red.text
