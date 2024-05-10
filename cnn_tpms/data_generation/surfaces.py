import numpy as np


def gyroid(x: float, y: float, z: float) -> float:
    return np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)


def schwarz_p(x: float, y: float, z: float) -> float:
    return np.cos(x) + np.cos(y) + np.cos(z)


def schwarz_d(x: float, y: float, z: float) -> float:
    a = np.sin(x) * np.sin(y) * np.sin(z)
    b = np.sin(x) * np.cos(y) * np.cos(z)
    c = np.cos(x) * np.sin(y) * np.cos(z)
    d = np.cos(x) * np.cos(y) * np.sin(z)
    return a + b + c + d


def neovius(x: float, y: float, z: float) -> float:
    a = 3 * (np.cos(x) + np.cos(y) + np.cos(z))
    b = 4 * np.cos(x) * np.cos(y) * np.cos(z)

    return a + b


def schoen_iwp(x: float, y: float, z: float) -> float:
    a = 2 * (np.cos(x) * np.cos(y) + np.cos(y) * np.cos(z) + np.cos(z) * np.cos(x))
    b = np.cos(2 * x) + np.cos(2 * y) + np.cos(2 * z)

    return a - b


def schoen_frd(x: float, y: float, z: float) -> float:
    a = 4 * np.cos(x) * np.cos(y) * np.cos(z)
    b = (
        np.cos(2 * x) * np.cos(2 * y)
        + np.cos(2 * y) * np.cos(2 * z)
        + np.cos(2 * z) * np.cos(2 * x)
    )
    return a - b


def fischer_koch_s(x: float, y: float, z: float) -> float:
    a = np.cos(2 * x) * np.sin(y) * np.cos(z)
    b = np.cos(x) * np.cos(2 * y) * np.sin(z)
    c = np.sin(x) * np.cos(y) * np.cos(2 * z)

    return a + b + c


def pmy(x: float, y: float, z: float) -> float:
    a = 2 * np.cos(x) * np.cos(y) * np.cos(z)
    b = np.sin(2 * x) * np.sin(y)
    c = np.sin(x) * np.sin(2 * z)
    d = np.sin(2 * y) * np.sin(z)

    return a + b + c + d


def honeycomb(x: float, y: float, z: float) -> float:
    return np.sin(x) * np.cos(y) + np.sin(y) + np.cos(z)
