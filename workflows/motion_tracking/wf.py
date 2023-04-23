import sieve

@sieve.workflow(name="point_tracking_klt")
def point_tracking_wf(vid: sieve.Video, x: int, y: int):
    return sieve.reference("sieve/klt_point_tracking")(vid, x, y)

@sieve.workflow(name="point_tracking_klt_superglue")
def point_tracking_superglue_wf(vid: sieve.Video, x: int, y: int):
    return sieve.reference("sieve/klt_superglue_point_tracking")(vid, x, y)

@sieve.workflow(name="point_tracking_klt_visualize")
def point_tracking_visualize_wf(vid: sieve.Video, x: int, y: int):
    tracked = sieve.reference("sieve/klt_point_tracking")(vid, x, y)
    consolidated = sieve.reference("sieve/consolidate_tracked_points")(tracked)
    return sieve.reference("sieve/point_tracking_visualize")(vid, consolidated), consolidated

@sieve.workflow(name="point_tracking_klt_superglue_visualize")
def point_tracking_superglue_visualize_wf(vid: sieve.Video, x: int, y: int):
    tracked = sieve.reference("sieve/klt_superglue_point_tracking")(vid, x, y)
    consolidated = sieve.reference("sieve/consolidate_tracked_points")(tracked)
    return sieve.reference("sieve/point_tracking_visualize")(vid, consolidated), consolidated
