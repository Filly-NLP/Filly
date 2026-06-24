from app.services.normalizer import get_normalizer

def test_dictionary_lookups():
    normalizer = get_normalizer()
    res = normalizer.normalize("aq po ay aalis na nmn")
    # should normalize: aq -> ako, nmn -> naman
    normalized_words = [n.word for n in res]
    suggestions = [n.suggestion for n in res]
    
    assert "aq" in normalized_words
    assert "nmn" in normalized_words
    assert "ako" in suggestions
    assert "naman" in suggestions
    
    # check exact properties
    aq_item = [n for n in res if n.word == "aq"][0]
    assert aq_item.confidence == 1.0
    assert aq_item.category == "slang"

def test_fuzzy_matching():
    normalizer = get_normalizer()
    res = normalizer.normalize("ak kc")
    # ak is within distance 1 of aq (which maps to ako)
    # kc is exact match for kasi
    normalized_words = [n.word for n in res]
    assert "ak" in normalized_words
    assert "kc" in normalized_words
    
    ak_item = [n for n in res if n.word == "ak"][0]
    assert ak_item.suggestion == "ako"
    assert ak_item.confidence == 0.7
    assert ak_item.category == "abbreviation"
