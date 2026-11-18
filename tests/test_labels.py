from orto_annotator.core.labels import LABEL_NAMES, Label


def test_three_primary_semantic_classes_exist():
    assert LABEL_NAMES[Label.SOCKET] == "Гильза"
    assert LABEL_NAMES[Label.CONNECTOR] == "Соединитель"
    assert LABEL_NAMES[Label.FOOT] == "Стопа"
