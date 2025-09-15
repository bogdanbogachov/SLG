import argparse
import json
import os
from openai import OpenAI

from config import CONFIG


if __name__ == '__main__':
    # Arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--create_qa", type=bool, default=False)
    parser.add_argument("--split_qa", type=bool, default=False)
    parser.add_argument("--inflate_overshadowing", type=bool, default=False)
    parser.add_argument("--combine_all_qa", type=bool, default=False)
    parser.add_argument("--data_overlap_check", type=bool, default=False)
    parser.add_argument("--evaluate", type=bool, default=False)
    parser.add_argument("--training_metrics", type=bool, default=False)
    parser.add_argument("--infer_slg", type=bool, default=False)
    parser.add_argument("--infer_baseline", type=bool, default=False)
    parser.add_argument("--infer_rag", type=bool, default=False)
    parser.add_argument("--infer_finetuned", type=bool, default=False)
    parser.add_argument("--download_models", type=bool, default=False)
    parser.add_argument("--finetune", type=bool, default=False)
    parser.add_argument("--charts", type=bool, default=False)
    args = parser.parse_args()

    # Download models
    if args.download_models:
        from download_llama import download, download_llama_3_1_8b
        download(model_name=CONFIG['3_2_1b'], save_directory='downloaded_3_2_1b')
        download_llama_3_1_8b(model_name=CONFIG['3_1_8b'], save_directory='downloaded_3_1_8b')

    # Question-answers
    if args.create_qa:
        from question_answer import populate
        from question_answer.srm_reader import read_doc
        from question_answer.om_reader import prepare_overhaul_manual

        # Create SRM QA pairs
        df_srm = read_doc('question_answer/srm.pdf')
        populate(df_srm, 'srm_qa')

        # Create OM QA pairs
        df_om = prepare_overhaul_manual(overhaul_manual='question_answer/om.pdf')
        populate(df_om, 'om_qa')

    if args.combine_all_qa:
        from question_answer import combine_all_qa
        combine_all_qa()


    if args.inflate_overshadowing:
        from question_answer import inflate_qa_answers_with_file_inputs
        inflate_qa_answers_with_file_inputs("question_answer/qa_original.json",
                                            "question_answer/inflating_material.json",
                                            "question_answer/qa.json")


    if args.split_qa:
        from question_answer import split_qa_pairs_by_title, split_train_test
        split_train_test('question_answer/qa.json')
        split_qa_pairs_by_title('question_answer/qa_train.json')

    # Measure data overlap
    if args.data_overlap_check:
        from evaluate import compute_overshadowing
        compute_overshadowing(prefix_length=5)

    # Experiments
    experiment = 'j_1_rag_3'
    # Finetune
    if args.finetune:
        from finetune import finetune
        os.makedirs('experiments', exist_ok=True)
        # Finetune SLG
        for file in os.listdir("question_answer/split_by_title"):
            finetune(
                model_to_tune='downloaded_3_2_1b',
                adapter_name=os.path.splitext(file)[0],
                data=f"question_answer/split_by_title/{file}",
                experiment_number=experiment,
                slg=True
            )

        # Finetune an orchestrator for SLG
        finetune(
            model_to_tune='downloaded_3_2_1b',
            adapter_name='orchestrator_3_2_1b',
            data='question_answer/qa_train.json',
            experiment_number=experiment,
            orchestrator=True
        )

        # Finetune llama 3.2 1B instruct
        finetune(
            model_to_tune='downloaded_3_2_1b',
            adapter_name='3_2_1b',
            data='question_answer/qa_train.json',
            experiment_number=experiment
        )

        # Finetune llama 3.1 8B instruct
        finetune(
            model_to_tune='downloaded_3_1_8b',
            adapter_name='3_1_8b',
            data='question_answer/qa_train.json',
            experiment_number=experiment
        )

    # Infer baseline
    if args.infer_baseline:
        from inference import ask_baseline
        client = OpenAI(api_key=CONFIG['open_ai_api_key'])
        os.makedirs(f'answers/{experiment}', exist_ok=True)
        ask_baseline(file='question_answer/qa_test.json', model=CONFIG['gpt_4_1'], experiment=experiment, client=client)

    if args.infer_rag:
        from inference import AskRag
        client = OpenAI(api_key=CONFIG['open_ai_api_key'])
        rag = AskRag(
            documents_file='question_answer/qa_train.json',
            questions_file='question_answer/qa_test.json',
            experiment=experiment,
            client=client
        )
        rag.generate_responses()

    if args.infer_finetuned:
        from inference import ask_finetuned
        os.makedirs(f'answers/{experiment}', exist_ok=True)
        ask_finetuned(file='question_answer/qa_test.json',
                      base_model='downloaded_3_2_1b',
                      adapter=f'experiments/{experiment}/finetuned_3_2_1b',
                      experiment=experiment)
        ask_finetuned(file='question_answer/qa_test.json',
                      base_model='downloaded_3_1_8b',
                      adapter=f'experiments/{experiment}/finetuned_3_1_8b',
                      experiment=experiment)

    # Infer slg
    if args.infer_slg:
        from inference import SmallLanguageGraph
        os.makedirs(f'answers/{experiment}', exist_ok=True)
        slg = SmallLanguageGraph(experts_location=experiment, experiment=experiment)
        slg.ask_slg(
            file='question_answer/qa_test.json'
        )

    # Evaluate
    if args.evaluate:
        from evaluate import load_data, evaluate, pull_training_metrics
        metrics_list = []
        ground_truth_file = "question_answer/qa_test.json"
        for predictions_file in os.listdir(f"answers/{experiment}"):
            predictions, ground_truth = load_data(f'answers/{experiment}/{predictions_file}',
                                                  ground_truth_file)
            results = evaluate(predictions, ground_truth)
            new_dict = {os.path.splitext(predictions_file)[0]: results}
            metrics_list.append(new_dict)
            os.makedirs(f"experiments/{experiment}", exist_ok=True)
            with open(f'experiments/{experiment}/metrics.json', 'w') as f:
                json.dump(metrics_list, f, indent=4)

        if args.training_metrics:
            # Add train and eval loss to
            training_metrics = pull_training_metrics(f'experiments/{experiment}')

            with open(f'experiments/{experiment}/metrics.json', "r") as f:
                data = json.load(f)  # Load JSON as a Python list

            data.extend(training_metrics)

            with open(f'experiments/{experiment}/metrics.json', "w") as f:
                json.dump(data, f, indent=4)

    # Build charts
    if args.charts:
        from evaluate import plot_finetuning_metrics

        experiment_root = 'experiments'

        param_folders = ['tune_lr_1', 'tune_lr_2', 'tune_lr_3']
        param_values = ['1e-5', '1e-4', '1e-3']
        plot_finetuning_metrics(experiment_root, param_folders, param_values, 'learning_rate')

        param_folders = ['tune_rank_1', 'tune_rank_2', 'tune_rank_3']
        param_values = [8, 16, 32]
        plot_finetuning_metrics(experiment_root, param_folders, param_values, 'lora_rank')

        param_folders = ['tune_grad_accum_1', 'tune_grad_accum_2', 'tune_grad_accum_3']
        param_values = [2, 4, 8]
        plot_finetuning_metrics(experiment_root, param_folders, param_values, 'gradient_accumulation_steps')

        param_folders = ['tune_alpha_1', 'tune_alpha_2', 'tune_alpha_3', 'tune_alpha_4']
        param_values = [8, 16, 32, 64]
        plot_finetuning_metrics(experiment_root, param_folders, param_values, 'lora_alpha')
