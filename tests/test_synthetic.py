import json

from catalog_cascade.synthetic import (
    COLORS,
    IMAGE_SIZE,
    SHAPES,
    TRAIN_SEED,
    data_url,
    make_items,
    save_items,
)


def test_items_are_reproducible_and_cover_every_combination():
    a = make_items(2, TRAIN_SEED)
    b = make_items(2, TRAIN_SEED)
    assert [(i.color, i.shape) for i in a] == [(i.color, i.shape) for i in b]
    assert len(a) == 2 * len(COLORS) * len(SHAPES)
    assert {(i.color, i.shape) for i in a} == {(c, s) for c in COLORS for s in SHAPES}


def test_target_is_exact_json_and_images_have_the_right_size():
    item = make_items(1, TRAIN_SEED)[0]
    assert json.loads(item.target) == {"color": item.color, "shape": item.shape}
    assert item.image.size == (IMAGE_SIZE, IMAGE_SIZE)
    assert data_url(item.image).startswith("data:image/png;base64,")


def test_shape_pixels_carry_the_colour():
    item = make_items(1, TRAIN_SEED)[0]
    assert item.image.getcolors(maxcolors=IMAGE_SIZE * IMAGE_SIZE)  # non-empty palette
    pixels = {rgb for _, rgb in item.image.getcolors(maxcolors=IMAGE_SIZE * IMAGE_SIZE)}
    assert COLORS[item.color] in pixels


def test_save_items_writes_images_and_manifest(tmp_path):
    items = make_items(1, TRAIN_SEED)[:3]
    manifest = save_items(items, tmp_path)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(rows) == 3
    assert all((tmp_path / row["image"]).exists() for row in rows)
