"""Analysis: per-frame pipeline, aggregation, statistics and export."""

from .comparison import friedman_test, position_comparison
from .export import export_results, write_markdown
from .frame_analysis import Segment10s, segment_frame, segment_frames
from .pipeline import FrameAnalyzer, ResultCache, run_analysis
from .statistics import (
    ACCEPTANCE_COLUMNS,
    ACCEPTED_CLASSES,
    FUSION_COLUMN,
    SQI_COLUMNS,
    acceptable_share,
    acceptance_recap,
    activity_recap,
    aggregate_by,
    box_outliers,
    class_distribution,
    class_table,
    describe,
    overall_summary,
    position_summary,
    results_to_frame,
    subject_position_matrix,
    subject_summary,
)

__all__ = [
    "ACCEPTANCE_COLUMNS",
    "ACCEPTED_CLASSES",
    "FUSION_COLUMN",
    "FrameAnalyzer",
    "ResultCache",
    "SQI_COLUMNS",
    "Segment10s",
    "acceptable_share",
    "acceptance_recap",
    "activity_recap",
    "aggregate_by",
    "box_outliers",
    "class_distribution",
    "class_table",
    "describe",
    "export_results",
    "friedman_test",
    "overall_summary",
    "position_comparison",
    "position_summary",
    "results_to_frame",
    "run_analysis",
    "segment_frame",
    "segment_frames",
    "subject_position_matrix",
    "subject_summary",
    "write_markdown",
]