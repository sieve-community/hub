import sieve

@sieve.workflow(name="pointrend_object_segmentation")
def segmentation(a: sieve.Video):
    m = sieve.reference("sieve/video-splitter")(a)
    ms = sieve.reference("sieve/pointrend_resnet50")(m)
    return sieve.reference("sieve/frame-combiner")(ms)
