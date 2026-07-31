"""Public API for the data-drift engine.

Everything a notebook, a batch job or the Streamlit app needs comes from one
import location::

    from drift import DriftAssessment, df_to_img

Two supporting modules are worth knowing about:

``drift.timeperiods``
    Understands a date column whatever its type -- text, numbers such as
    ``201903``, quarter labels, timestamps, pandas ``Period`` -- and cuts it into
    snapshots. **Monthly is the default.**
``drift.storage``
    Decides where results are saved (``results/drift/`` beside this repository by
    default, redirected with :func:`set_output_dir`) and writes every file in a
    single operation, so a local folder, DBFS and ADLS all behave the same.
"""

from .drift_assessment import (
    DriftAssessment,
    MyException,
    merge_one_hot_encoded_columns,
    psi_coloring,
    yes_no,
)
from .drift_util import (
    ALERT_COLOR,
    HEADER_COLOR,
    PSI_THRESHOLDS,
    ROW_COLORS,
    WARN_COLOR,
    df_to_img,
)
from .storage import (
    default_output_dir,
    join_address,
    output_dir,
    reset_output_dir,
    set_output_dir,
)
from .timeperiods import (
    DEFAULT_INTERVAL,
    INTERVALS,
    normalise_interval,
    parse_time_column,
    period_labels,
    snapshot_labels,
)

__version__ = "2.1.0"

__all__ = [
    "__version__",
    # assessment
    "DriftAssessment",
    "MyException",
    "merge_one_hot_encoded_columns",
    "psi_coloring",
    "yes_no",
    # rendering
    "ALERT_COLOR",
    "HEADER_COLOR",
    "PSI_THRESHOLDS",
    "ROW_COLORS",
    "WARN_COLOR",
    "df_to_img",
    # where results are saved
    "default_output_dir",
    "join_address",
    "output_dir",
    "reset_output_dir",
    "set_output_dir",
    # time handling
    "DEFAULT_INTERVAL",
    "INTERVALS",
    "normalise_interval",
    "parse_time_column",
    "period_labels",
    "snapshot_labels",
]
