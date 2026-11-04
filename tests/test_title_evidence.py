from annotation.contracts import Box, Evidence, FieldClaim, OCRResult, OCRSpan


def span(text, rotation=0, confidence=0.9, y=0.1):
    return OCRSpan(
        text=text,
        rotation=rotation,
        confidence=confidence,
        box=Box(x0=0.1, y0=y, x1=0.8, y1=y + 0.05),
    )


def claim(text):
    return FieldClaim(
        status="present", value=text, evidence=[Evidence(quote=text)], reason="visible"
    )


def test_title_matches_punctuation_and_adjacent_lines_within_one_rotation():
    from annotation.title_evidence import title_support

    ocr = OCRResult(
        spans=[span("JURISPRUDENCE"), span("T", confidence=0.1), span("< LEGAL THEORY", y=0.17)]
    )
    assert title_support(claim("Jurisprudence & Legal Theory"), ocr, 0.5)["supported"]


def test_title_never_joins_rotations_or_changes_words():
    from annotation.title_evidence import title_support

    assert not title_support(
        claim("Administrative Tribunals Act 1985"),
        OCRResult(spans=[span("Administrative", 0), span("Tribunals Act 1985", 180)]),
        0.5,
    )["supported"]
    ocr = OCRResult(spans=[span("The Administrative Tribunals Act, 1985")])
    assert not title_support(claim("The Administrator's Act 1985"), ocr, 0.5)["supported"]
    ocr = OCRResult(
        spans=[span("Blackstone's Statutes on"), span("Contract, Tort & Restitution", y=0.17)]
    )
    assert not title_support(
        claim("Blackstone's Statutes & Contract, Tort & Restitution"), ocr, 0.5
    )["supported"]


def test_title_does_not_bridge_far_apart_books():
    from annotation.title_evidence import title_support

    ocr = OCRResult(spans=[span("Alpha", y=0.1), span("Beta", y=0.8)])
    assert not title_support(claim("Alpha Beta"), ocr, 0.5)["supported"]
