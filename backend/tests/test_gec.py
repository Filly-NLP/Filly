from app.services.gec import get_gec_service

def test_repeated_words():
    gec = get_gec_service()
    res = gec.check("ang ang bata ay sumayaw")
    assert len(res) == 1
    assert res[0].original == "ang ang"
    assert res[0].correction == "ang"
    assert res[0].rule == "repeated_word"

def test_din_rin_rule():
    gec = get_gec_service()
    # Ends in vowel, din -> rin
    res = gec.check("bata din")
    assert len(res) == 1
    assert res[0].original == "din"
    assert res[0].correction == "rin"
    
    # Ends in consonant, remains din
    res2 = gec.check("kain din")
    assert len(res2) == 0

def test_daw_raw_rule():
    gec = get_gec_service()
    # Ends in vowel, daw -> raw
    res = gec.check("sabi daw")
    assert len(res) == 1
    assert res[0].original == "daw"
    assert res[0].correction == "raw"
    
    # Ends in consonant, remains daw
    res2 = gec.check("alis daw")
    assert len(res2) == 0

def test_ng_nang_rule():
    gec = get_gec_service()
    res = gec.check("kumain nang pagkain")
    assert len(res) == 1
    assert res[0].original == "nang pagkain"
    assert res[0].correction == "ng pagkain"
    assert res[0].rule == "ng_nang"
