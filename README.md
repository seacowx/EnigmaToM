# EnigmaToM
This repository contains the code and data for EnigmaToM. We are still in the process of cleaning up the code and will update this README file constantly as we progress.

# TODOs
- [] Clean up the code and add comments
- [] Add more details to the README file
- [] Add requirements.txt file
- [] Add script for easy inference 
    - [] Allow user to specify different models
    - [] Allow user to specify different datasets

<!-- # Data Usage -->
<!-- --- -->
<!-- All ToM datasets are stored in `./data/tom_datasets/`. -->
<!-- All ToM datasets with reduced ToM questions are stored with a `_reduced_q` postfix.  -->
<!---->
<!---->
<!-- # Code Usage -->
<!-- --- -->
<!-- To generate MaskToM states, use the following command: -->
<!-- ```bash -->
<!-- python generate_states.py --data_path PATH_TO_DATA --model MODEL_NAME --quantization BIT -->
<!-- ``` -->
<!-- The generated results are saved at `./data/masktom/` -->
<!---->
<!-- To generate the world states, use the following command: -->
<!-- ```bash -->
<!-- python generate_states.py --data_path PATH_TO_DATA --model MODEL_NAME --quantization BIT --world_state -->
<!-- ``` -->
<!-- The generated results are saved at `./data/masktom/world_state/MODEL_NAME/` -->
<!---->
<!-- --- -->
<!---->
<!-- To reduce the ToM question, use the following command: -->
<!-- ```bash -->
<!-- python reduce_tom_questions.py --data_path PATH_TO_DATA -->
<!-- ``` -->
<!---->
<!-- --- -->
<!---->
<!-- To run experiments, use the following command: -->
<!-- ```bash -->
<!-- python run_exps.py --data_path PATH_TO_DATA_WITH_REDUCED_TOM_QUESTIONS --model MODEL_NAME -->
<!-- ``` -->
<!-- configs to the experiments are located in `./src/configs/experiment_config.yml`, `./src/configs/hf_config.yml`, and `./src/configs/model_info.yml` -->
