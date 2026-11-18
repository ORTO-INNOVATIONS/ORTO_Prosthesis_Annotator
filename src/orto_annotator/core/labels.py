"""Semantic label definitions and display colors."""

from enum import IntEnum


class Label(IntEnum):
    UNLABELED = 0
    SOCKET = 1
    CONNECTOR = 2
    FOOT = 3
    UNCERTAIN = 4


LABEL_NAMES = {
    Label.UNLABELED: "Не размечено",
    Label.SOCKET: "Гильза",
    Label.CONNECTOR: "Соединитель",
    Label.FOOT: "Стопа",
    Label.UNCERTAIN: "Неопределённо",
}

# RGB values are used consistently in the 3D viewport and exported metadata.
LABEL_COLORS = {
    Label.UNLABELED: (0.72, 0.74, 0.77),
    Label.SOCKET: (0.16, 0.45, 0.82),
    Label.CONNECTOR: (0.91, 0.61, 0.17),
    Label.FOOT: (0.13, 0.64, 0.47),
    Label.UNCERTAIN: (0.63, 0.42, 0.72),
}
