🔄 Setting up lm_eval wrapper...
[lm_eval.models.huggingface|WARNING]`pretrained` model kwarg is not of type `str`. Many other model arguments may be ignored. Please do not launch via accelerate or use `parallelize=True` if passing an existing model this way.


❌ Error: 
Traceback (most recent call last):
  File "/workspace/scripts/evaluate_model.py", line 591, in <module>
    main()
  File "/workspace/scripts/evaluate_model.py", line 565, in main
    results = run_evaluation(model, tokenizer, selected_tasks, limit)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/scripts/evaluate_model.py", line 366, in run_evaluation
    llm = HFLM(
          ^^^^^
  File "/workspace/Automodel/.venv/lib/python3.12/site-packages/lm_eval/models/huggingface.py", line 204, in __init__
    self._create_tokenizer(
  File "/workspace/Automodel/.venv/lib/python3.12/site-packages/lm_eval/models/huggingface.py", line 778, in _create_tokenizer
    assert isinstance(
           ^^^^^^^^^^^
AssertionError