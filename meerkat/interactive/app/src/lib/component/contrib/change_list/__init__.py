from ...abstract import BaseComponent


class ChangeList(BaseComponent):
    gallery: BaseComponent
    gallery_match: BaseComponent
    gallery_filter: BaseComponent
    gallery_sort: BaseComponent
    discover: BaseComponent
    plot: BaseComponent
    active_slice: BaseComponent
    slice_sort: BaseComponent
    slice_match: BaseComponent
    global_stats: BaseComponent
