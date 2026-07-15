__all__ = [
    "CQADUPSTACK_SUBFORUMS",
    "DEFAULT_CQADUPSTACK_URL",
    "convert_cqadupstack_clusters_to_qa",
    "convert_cqadupstack_physics_to_qa",
    "download_cqadupstack",
    "ensure_stackexchange_posts_xml",
    "filter_qa_by_answer_question_count",
]


def __getattr__(name):
    if name == "CQADUPSTACK_SUBFORUMS":
        from question_answer.download_cqadupstack import CQADUPSTACK_SUBFORUMS

        return CQADUPSTACK_SUBFORUMS
    if name == "DEFAULT_CQADUPSTACK_URL":
        from question_answer.download_cqadupstack import DEFAULT_CQADUPSTACK_URL

        return DEFAULT_CQADUPSTACK_URL
    if name == "download_cqadupstack":
        from question_answer.download_cqadupstack import download_cqadupstack

        return download_cqadupstack
    if name == "convert_cqadupstack_physics_to_qa":
        from question_answer.cqadupstack_physics_converter import convert_cqadupstack_physics_to_qa

        return convert_cqadupstack_physics_to_qa
    if name == "convert_cqadupstack_clusters_to_qa":
        from question_answer.cqadupstack_clustered_qa_converter import convert_cqadupstack_clusters_to_qa

        return convert_cqadupstack_clusters_to_qa
    if name == "ensure_stackexchange_posts_xml":
        from question_answer.cqadupstack_clustered_qa_converter import ensure_stackexchange_posts_xml

        return ensure_stackexchange_posts_xml
    if name == "filter_qa_by_answer_question_count":
        from question_answer.filter_qa_by_answer_question_count import filter_qa_by_answer_question_count

        return filter_qa_by_answer_question_count
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
