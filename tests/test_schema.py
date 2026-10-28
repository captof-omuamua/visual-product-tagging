from catalog_cascade import schema


def test_synonyms_and_first_word_wins():
    assert schema.normalize("color", "SOFA", "Grey Linen") == "grey"
    assert schema.normalize("color", "SHOES", "Microsuede Cognac") == "brown"
    assert schema.normalize("material", "SOFA", "velvet, hardwood frame, metal legs") == "fabric"
    assert schema.normalize("style", "CHAIR", "Mid-Century Modern") == "mid-century"
    assert schema.normalize("style", "SHOES", "Running Shoes") == "athletic"


def test_no_match_gives_no_vote():
    assert schema.normalize("material", "SHOES", "Synthetic") is None
    assert schema.normalize("color", "RUG", None) is None
    assert schema.normalize("color", "RUG", "Redwood") is None  # whole words only


def test_type_fields():
    assert "shape" in schema.fields_for("RUG")
    assert "finish" in schema.fields_for("LAMP")
    assert set(schema.fields_for("SHOES")) == {"color", "material", "pattern", "style"}
    enum = schema.json_schema("TABLE")["properties"]["shape"]["enum"]
    assert enum[-2:] == ["other", "none"]
