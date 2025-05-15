# testing all sorts of functions for my own sanity

import numpy as np
from src.zone_detect.test.metrics import error_rate_patch


def test_error_rate(img_path: str, out_dir: str, verbose: bool = False) -> None:

    # for two identical images, the error rate should be 0 everywhere

    print(
        """Test for error_rate_patch : 
    - Testing identical images"""
    )

    error = error_rate_patch(img_path, out_dir, img_path)

    if not np.any(error):
        print("     [x] OK")
    else:
        if verbose:
            print(
                f"""     [ ] ERROR
                The output looks like this : {error[:6, :6]}"""
            )
        print(
            f"""     [ ] ERROR
        Go check the output at : {out_dir}"""
        )
