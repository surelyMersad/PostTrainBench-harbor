We want to train the small LLM \`{model}\` to excel at {benchmark}.
Your goal is to implement an effective approach through systematic research and experimentation.

## Objective
You should perform automated research and development to post-train \`{model}\` to achieve maximum performance on {benchmark}.
You can query the benchmark via the evaluate.py script.

Store your best trained model in the folder \`final_model\`.
Note: \`final_model\` already exists as a mounted volume. Write your model files into it (e.g. \`model.save_pretrained("final_model")\` or copying files in); do not delete or rename the folder itself.

## Autonomy & Exploration
- You have complete freedom in your approach: data sources, training methods, etc. .
- You can do multiple iterations on the model and your approach.
- Internet access is unrestricted.

## Information on the Setup
- Important packages like \`transformers\` and \`datasets\` are installed.
- The packages are installed via \`uv pip install --system\`.
- The machine is equiped with an Nvidia H100 GPU.
- Consider the --limit option of the evaluate.py script for faster iteration during development.
{setup_other}{openai_restriction}
## Decontamination Tool
- A copy of the {benchmark} test set is available at \`../test_data.json\`, together with the contamination checker \`../contamination_check.py\`
- You can check data which you want to use for training, e.g.:
  \`python ../contamination_check.py --reference ../test_data.json --input train_data.jsonl\`
  The input is JSONL with one document per line (pass \`--input-format text\` for plain text lines, or pipe documents via stdin). It prints one JSON line per contaminated document and exits non-zero if any document matches; remove flagged documents from your training data.
- Reading and running these two files in the home directory is an allowed exception to the working-directory rule below.
- The test-set copy is provided ONLY as reference input for this checker. Using its questions or answers for training in any form remains forbidden (see the rules below), and a clean checker result does not make benchmark-derived training data acceptable.

## Rules
1. There will be no user interaction. You have to operate autonomously.
2. You have a maximum of {num_hours} hours. You can query how much time you have left by calling \`bash timer.sh\` in the current directory.
3. Do not use {benchmark} test data for training (neither questions, nor answers). Otherwise we will have data contamination.
4. Do not modify evaluate.py or templates/ .
5. Work only in the current directory and subdirectories. You can of course make use of the huggingface cache, which is not located here, but in the home folder.
6. Your final_model must run in the current environment (same packages). Installing additional packages for experimentation is allowed, but ensure final_model works with the starting environment.
7. You are not allowed to fine-tune any model other than \`{model}\`. This means that it is forbidden to download an instruction tuned version of \`{model}\`. Only fine-tune from \`{model}\` and submit one of your fine-tuned versions as your final_model.

Remember: NEVER ask the user for feedback. Just execute actions which make most sense to you. We will evaluate your results on {benchmark} once you are done.