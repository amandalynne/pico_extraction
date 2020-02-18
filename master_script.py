import json
import random
import subprocess
import tempfile

from itertools import product

def main(env):
    """
    This script launches the following experiments on Beaker:
        1. PICO sentence classification
        2. Label induction from attn weights from (1)
        3. Initialize token labeler with parameters from (1)
        4. Train token labeler on induced labels from (2)
        5. Finetune token labeler from (4)
    
    The specs for these experiments are in two AllenNLP configs in allennlp_configs/: 
    'pico_multi_attn.jsonnet' and 'pico_crf_tagger.jsonnet', both of which are
    configurable using different hyperparameters specified as environment variables.

    Parameters
    -----------
    env: a dictionary of environment variables to use in grid search.
    """

    # get a Git SHA to uniquely identify current state of the code
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], universal_newlines=True).strip()
    # use first 7 digits
    image_id = git_sha[:7]

    # Check to see if the image already exists
    docker_image_command = f'docker images {image_id} --format {{{{.Repository}}}}'

    if subprocess.check_output(docker_image_command, shell=True, universal_newlines=True):
        # Image exists
        pass
    else:
        # Make the image
        subprocess.run(f'docker build -t {image_id} .', shell=True, check=True)

    # make Beaker blueprint
    blueprint = subprocess.check_output(f'beaker blueprint create --quiet {image_id}', shell=True, universal_newlines=True).strip() 

    # Requirements
    reqs = { "gpuCount": 1 }

    # make Beaker spec for 5 tasks 
    # Task 1: sentence classifier
    attn_config = 'allennlp_config/pico_multi_attn.jsonnet'
    upload_config = f'beaker dataset create --quiet {attn_config}'
    config_dataset_id = subprocess.check_output(upload_config, shell=True, universal_newlines=True).strip()
    allennlp_command = [
        "allennlp",
        "train",
        "-s",
        "/output",
        f"/{attn_config}",
        "--include-package",
        "pico_extraction",
        "--overrides",
        f'{{"trainer": {{ "optimizer": {{"lr": {env["learning_rate"]}, "weight_decay": {env["weight_decay"]} }} }} }}'
    ]

    task_1 = { 
        "name": "sentence_classify", 
        "spec": {
            "blueprint": blueprint,
            "resultPath": "/output",
            "args": allennlp_command,
            "datasetMounts": [
                { "datasetId": config_dataset_id,
                "containerPath": f"/{attn_config}" },
                { "datasetId": "REDACTED",
                "containerPath": "/sentence_data" },
                { "datasetId": "REDACTED",
                "containerPath": "/elmo" }
            ], 
            "requirements": reqs,
            "env": {
                "TRAIN_DATA_PATH": "/sentence_data/PICO_train.txt",
                "DEV_DATA_PATH": "/sentence_data/PICO_dev.txt",
                "BATCH_SIZE": env['sent_batch_size'],
                "EPOCHS": env['epochs'], 
                "GAMMA": env['gamma'],
            }
        }
    } 
    
    python_command = [
        "python",
        "induce_labels.py",
        "/sentence_classify/model.tar.gz",
        "/PICO_train.txt",
        "/output/induced_labels.txt"
    ] 

    # Task 2: label induction
    task_2 = { 
        "name": "label_induction", 
        "spec": {
            "blueprint": blueprint,
            "args": python_command,
            "resultPath": "/output",
            "datasetMounts": [
                { "datasetId": "REDACTED",
                  "containerPath": "/PICO_train.txt"
                }
            ],
            "requirements": reqs
        },
        "dependsOn": [ 
            { "parentName": "sentence_classify",
            "containerPath": "/sentence_classify" }
        ]
    } 


    # Task 3: pretrain
    crf_tag_config = "allennlp_config/pico_crf_tagger.jsonnet" 
    upload_config = f'beaker dataset create --quiet {crf_tag_config}'
    config_dataset_id = subprocess.check_output(upload_config, shell=True, universal_newlines=True).strip()

    pretrain_command = [
        "allennlp",
        "train",
        "-s",
        "/output",
        f"/{crf_tag_config}",
        "--include-package",
        "pico_extraction"
    ]

    task_3 = {
        "name": "token_transfer", 
        "spec": {
            "blueprint": blueprint,
            "args": pretrain_command,
            "resultPath": "/output",
            "datasetMounts": [ 
                { "datasetId": config_dataset_id,
                "containerPath": f"/{crf_tag_config}" },
                { "datasetId": "REDACTED", 
                "containerPath": "/io_tags" },
                { "datasetId": "REDACTED",
                "containerPath": "/elmo" }
            ],
            "requirements": reqs,
            "env": {
                "PRETRAINED_MODEL_PATH": "None",
                "TRAIN_DATA_PATH": "/label_induction",
                "DEV_DATA_PATH": "/io_tags/dev",
                "BATCH_SIZE": env['pretrain_batch_size'],
                "EPOCHS": env['epochs']
            }
        },
        "dependsOn": [
            {
            "parentName": "label_induction",
            "containerPath": "/label_induction"
            }
        ]
    }

    # Task 4: finetune 
    finetune_command = [
        "allennlp",
        "train",
        "-s",
        "/output",
        f"/{crf_tag_config}",
        "--include-package",
        "pico_extraction"
    ]

    task_4 = {
        "name": "finetune_token",
        "spec": {
            "blueprint": blueprint,
            "args": finetune_command,
            "resultPath": "/output",
            "datasetMounts": [
                { "datasetId": config_dataset_id,
                "containerPath": f'/{crf_tag_config}' },
                { "datasetId": "REDACTED", 
                "containerPath": "/io_tags" },
                { "datasetId": "REDACTED",
                "containerPath": "/elmo" }
            ], 
            "requirements": reqs,
            "env": {
                "PRETRAINED_MODEL_PATH": "/token_transfer/model.tar.gz",
                "TRAIN_DATA_PATH": "/io_tags/train",
                "DEV_DATA_PATH": "/io_tags/dev",
                "BATCH_SIZE": env['finetune_batch_size'],
                "EPOCHS": env['epochs']
            }
        },
        "dependsOn": [
            {
            "parentName": "token_transfer",
            "containerPath": "/token_transfer"
            }
        ]
    }

    # Task 5: initialize EBM-NLP with sentence classifier
    sentence_command = [
        "allennlp",
        "train",
        "-s",
        "/output",
        f"/{crf_tag_config}",
        "--include-package",
        "pico_extraction"
    ]

    task_5 = {
        "name": "sentence_transfer",
        "spec": {
            "blueprint": blueprint,
            "args": sentence_command,
            "resultPath": "/output",
            "datasetMounts": [
                { "datasetId": config_dataset_id,
                "containerPath": f'/{crf_tag_config}' },
                { "datasetId": "REDACTED", 
                "containerPath": "/io_tags" },
                { "datasetId": "REDACTED",
                "containerPath": "/elmo" }
            ], 
            "requirements": reqs,
            "env": {
                "PRETRAINED_MODEL_PATH": "/sentence_classify/model.tar.gz",
                "TRAIN_DATA_PATH": "/io_tags/train",
                "DEV_DATA_PATH": "/io_tags/dev",
                "BATCH_SIZE": env['finetune_batch_size'],
                "EPOCHS": env['epochs']
            }
        },
        "dependsOn": [
            {
            "parentName": "sentence_classify",
            "containerPath": "/sentence_classify"
            }
        ]
    }
    
    tasks = [task_1, task_2, task_3, task_4, task_5]
    beaker_spec = {
        "description": str(env), 
        "tasks": tasks 
    }

    # make temporary yaml
    output_path = tempfile.mkstemp(".yaml", "temp")[1]
    with open(output_path, "w") as output:
        output.write(json.dumps(beaker_spec, indent=4))
    # launch experiment on Beaker
    subprocess.run(f'beaker experiment create -f {output_path}', shell=True)

def dict_product(d):
    """
    Helpful for doing grid search over parameters.
    Returns a generator over the cross product of values in a dictionary, retaining keys.
    For example, if the input is { 'batch_size': [1, 2], 'epochs': [3] },
    the output is a generator over:
    { 'batch_size': 1, 'epochs': 3 }
    { 'batch_size': 2, 'epochs': 3 }
    """
    return (dict(zip(d.keys(), values)) for values in product(*d.values()))


if __name__ == "__main__":
    # Hyperparameters to try
    hyperparameters = { 
        'sent_batch_size': [40],
        'pretrain_batch_size': [20, 40],
        'finetune_batch_size': [20, 40],
        'weight_decay': [0, 1e-3, 1e-5],
        'learning_rate': [0.01, 0.001],
        'epochs': [15],
        'gamma': [0.01, 0.1, 1]
    }
    # Generate combinations of parameters and then run main.
    for env in dict_product(hyperparameters):
        # Randomly select only some of the hyperparameter combos.
        rand = random.randint(0,100)
        if rand % 3 == 0:
            main(env)
