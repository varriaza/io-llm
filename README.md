# io-llm

This project is a collection of AI models that can be run interactively. The models are designed to generate responses to user input in a conversational manner. The models are configured using YAML files, and the project includes several pre-configured models such as `mistral_7b_instruct`, `snorkel`, and `truthful`.

## Setup and Installation
1. Clone the repo 
```sh
git clone https://github.com/varriaza/io-llm.git
```
2. In the terminal run:
```sh
python -m venv io-llm
source io-llm/bin/activate
```
3. In the terminal run:
```sh
pip install -r requirements.txt
```
4. You are done!

## Running the Project

To run a model, you need to execute the `run.py` script with the YAML configuration file for the model as an argument. 

### mistral_7b_instruct
```sh
python run.py mistral_7b
```

### truthful
```sh
python run.py truthful
```

### snorkel
```sh
python run.py snorkel 
```

When you run this command, the script will load the model specified in the YAML file and start an interactive chat session. You can type your input at the prompt, and the model will generate a response.

To stop the interactive session, type `exit`.

## Models

The models are implemented as Python classes in the models directory. Each model class inherits from the `BaseModel` class in `models/base_model.py` and implements the `setup_model` and `run_model` methods.

The `setup_model` method sets up the model for conversation, and the `run_model` method generates a response to a given input text.

## Configuration

The YAML configuration files in the `config_files` directory specify the settings for each model. These settings include the model name, class name, and various parameters related to the model's operation.

For example, the `mistral_7b.yml` file configures the `mistral_7b_instruct` model. The `model` field specifies the model name, the `class_name` field specifies the class name, and the `repo_id` field specifies the repository ID for the model on the Hugging Face.
