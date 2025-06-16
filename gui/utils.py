import base64

import numpy as np


def base64_to_numpy_array(b64_string, dtype=np.uint8) -> np.ndarray:
    decoded = base64.b64decode(b64_string)
    return np.frombuffer(decoded, dtype=dtype)
