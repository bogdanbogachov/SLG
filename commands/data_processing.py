import os
from logging_config import logger
from config import CONFIG


def create_qa():
    from question_answer.question_answer_gen import populate
    from question_answer.srm_reader import read_doc
    from question_answer.om_reader import prepare_overhaul_manual

    files_config = CONFIG['files']
    srm_pdf = files_config['srm_pdf']
    om_pdf = files_config['om_pdf']
    
    df_srm = read_doc(srm_pdf)
    populate(df_srm, 'srm_qa')

    df_om = prepare_overhaul_manual(overhaul_manual=om_pdf)
    populate(df_om, 'om_qa')


def combine_all():
    from question_answer.question_answer_gen import combine_all_qa
    combine_all_qa()


def inflate_overshadowing():
    from question_answer.inflate_overshadowing import inflate_qa_answers_with_file_inputs
    files_config = CONFIG['files']
    inflate_qa_answers_with_file_inputs(
        files_config['qa_original'],
        files_config['qa_inflating_material'],
        files_config['qa']
    )


def split_qa():
    from question_answer.question_answer_gen import split_qa_pairs_by_title, split_train_test
    files_config = CONFIG['files']
    split_train_test(files_config['qa'])
    split_qa_pairs_by_title(files_config['qa_train'])


def data_overlap_check():
    from evaluate.data_overlap import compute_overshadowing
    data_config = CONFIG['data']
    prefix_length = data_config['prefix_length']
    compute_overshadowing(prefix_length=prefix_length)
