# torchtune

**Start:** 2024-04-16
**End:** TODO

- [ ] Code: <https://github.com/pytorch/torchtune>
  - [ ] torchtune lib: pytorch/torchtune/torchtune/
    - [X] _cli
    - [X] config
    - [ ] data
    - [X] datasets
    - [ ] dev
    - [X] generation
    - [ ] models
      - [X] llama2
      - [ ] llama3 - understand tokenizer more deeply
      - [X] llama3_1
      - [X] llama3_2
      - [ ] llama3_2_vision - understand if this approach is applicable for EMG?
      - [X] llama3_3
    - [X] modules
      - [X] modules/*.py
      - [X] modules/_export
      - [X] modules/loss
      - [X] modules/low_precision
      - [X] modules/model_fusion
      - [X] modules/peft
      - [X] modules/tokenizers
      - [X] modules/transforms
    - [ ] rlhf
    - [X] training
    - [X] utils
    - [X] recipe_interfaces.py
    - [X] _recipe_registry.py
  - [ ] recipes: pytorch/torchtune/recipes
    - [ ] recipe/configs skim
    - [X] generate.py
    - [X] full_finetune_single_device.py
    - [X] full_finetune_distributed.py
    - [X] lora_finetune_single_device.py
    - [X] lora_finetune_distributed.py
    - [ ] TODO
