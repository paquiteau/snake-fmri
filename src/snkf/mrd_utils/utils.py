"""Utils for the MRD format."""

import base64
import pickle
from enum import IntFlag
from typing import Any


def obj2b64encode(f: Any) -> bytes:
    """Return the base64 encoded pickle of a python object."""
    return base64.b64encode(pickle.dumps(f))


def b64encode2obj(s: str) -> Any:
    """Load a base64 string as a python object."""
    return pickle.loads(base64.b64decode(s))


# fmt: off
class ACQ(IntFlag):
    """Acquisition flags of MRD as an IntFlags."""

    FIRST_IN_ENCODE_STEP1               = 1 << 0
    LAST_IN_ENCODE_STEP1                = 1 << 1
    FIRST_IN_ENCODE_STEP2               = 1 << 2
    LAST_IN_ENCODE_STEP2                = 1 << 3
    FIRST_IN_AVERAGE                    = 1 << 4
    LAST_IN_AVERAGE                     = 1 << 5
    FIRST_IN_SLICE                      = 1 << 6
    LAST_IN_SLICE                       = 1 << 7
    FIRST_IN_CONTRAST                   = 1 << 8
    LAST_IN_CONTRAST                    = 1 << 9
    FIRST_IN_PHASE                      = 1 << 10
    LAST_IN_PHASE                       = 1 << 11
    FIRST_IN_REPETITION                 = 1 << 12
    LAST_IN_REPETITION                  = 1 << 13
    FIRST_IN_SET                        = 1 << 14
    LAST_IN_SET                         = 1 << 15
    FIRST_IN_SEGMENT                    = 1 << 16
    LAST_IN_SEGMENT                     = 1 << 17
    IS_NOISE_MEASUREMENT                = 1 << 18
    IS_PARALLEL_CALIBRATION             = 1 << 19
    IS_PARALLEL_CALIBRATION_AND_IMAGING = 1 << 20
    IS_REVERSE                          = 1 << 21
    IS_NAVIGATION_DATA                  = 1 << 22
    IS_PHASECORR_DATA                   = 1 << 23
    LAST_IN_MEASUREMENT                 = 1 << 24
    IS_HPFEEDBACK_DATA                  = 1 << 25
    IS_DUMMYSCAN_DATA                   = 1 << 26
    IS_RTFEEDBACK_DATA                  = 1 << 27
    IS_SURFACECOILCORRECTIONSCAN_DATA   = 1 << 28
    IS_PHASE_STABILIZATION_REFERENCE    = 1 << 29
    IS_PHASE_STABILIZATION              = 1 << 30
    COMPRESSION1                        = 1 << 52
    COMPRESSION2                        = 1 << 53
    COMPRESSION3                        = 1 << 54
    COMPRESSION4                        = 1 << 55
    USER1                               = 1 << 56
    USER2                               = 1 << 57
    USER3                               = 1 << 58
    USER4                               = 1 << 59
    USER5                               = 1 << 60
    USER6                               = 1 << 61
    USER7                               = 1 << 62
    USER8                               = 1 << 63
# fmt: on
