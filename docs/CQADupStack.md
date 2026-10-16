# CQADupStack data provenance

CQAD was prepared from the public [BEIR CQADupStack dataset](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip) and Stack Exchange post dumps. The repository scripts perform the download and conversion; the experiment dataset builder consumes the already-created `question_answer/cqadupstack_clustered/qa_min20.json` and does not download anything.

## Download and conversion

1. `question_answer/download_cqadupstack.py` downloads the BEIR ZIP and extracts the 12 CQADupStack communities under `data/cqadupstack/`. Each community contains `corpus.jsonl`, `queries.jsonl`, and `qrels/test.tsv`. By default the ZIP archive is deleted after successful extraction. The resulting `manifest.json` records the source URL and extracted communities.

   ```bash
   python question_answer/download_cqadupstack.py
   ```

2. `question_answer/cqadupstack_clustered_qa_converter.py` turns the duplicate-question links into question clusters, then joins their post IDs against each community's Stack Exchange `Posts.xml`. For this it maps each CQAD community to its Stack Exchange site and downloads the corresponding `.7z` post dump from `https://archive.org/download/stackexchange/` when `--download-missing` is supplied. It extracts `Posts.xml`; downloaded `.7z` archives are deleted after extraction unless `--keep-archives` is supplied. Each retained cluster gets one canonical answer, preferring an accepted answer and otherwise using the converter's score-based selection. The default minimum cluster size is 4.

   ```bash
   python question_answer/cqadupstack_clustered_qa_converter.py --download-missing --min-cluster-size 4
   ```

   This writes `question_answer/cqadupstack_clustered/qa.json`, `clusters.csv`, and `conversion_report.json`.

3. `question_answer/filter_qa_by_answer_question_count.py` keeps answer groups with at least 20 distinct questions, grouped within each community (`chapter`). This produced `qa_min20.json` and its report:

   ```bash
   python question_answer/filter_qa_by_answer_question_count.py \
     --input-file question_answer/cqadupstack_clustered/qa.json \
     --output-file question_answer/cqadupstack_clustered/qa_min20.json \
     --min-questions-per-answer 20
   ```

## Current recorded counts

The checked-in conversion report records 11,697 QA rows from 1,467 clusters across all 12 communities. The min-20 report records 3,435 rows from 72 answer groups. The experiment `cqad20` is built from `qa_min20.json`.

## Storage cleanup

After conversion, the experiment build only needs `question_answer/cqadupstack_clustered/qa_min20.json`. The 15 GB `data/cqadupstack/` download is an intermediate BEIR input and can be removed once no further conversion or provenance re-check is needed. If `data/stackexchange/` exists, its `Posts.xml` files and any `_archives/*.7z` are also conversion inputs; deleting them means rerunning conversion later will require downloading the post dumps again. Keep the small `question_answer/cqadupstack_clustered/` outputs and reports if you want to retain the generated source and its audit trail.
