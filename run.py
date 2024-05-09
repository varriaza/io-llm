import yaml
import sys
from icecream import ic
from huggingface_hub import hf_hub_download
from importlib import import_module

def load_variables(yaml_file) -> dict[str, any]:
    """
    Load the variables from the yaml file.
    """
    # Add .yml extension if not present
    if not yaml_file.endswith(".yml"):
        yaml_file += ".yml"

    # Try to load the yaml file
    try:
        with open("config_files/" + yaml_file, "r") as file:
            variables = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"File {yaml_file} not found in config_files/. Please check your spelling!"
        )
    return variables


def main(yaml_file):
    """
    Run the model specified in the yaml file.
    """
    # Load variables and standardize to lowercase
    variables = load_variables(yaml_file)
    model_name = variables["model"].lower()
    class_name = variables["class_name"]

    # Dynamically import the model class
    module = import_module(f"models.{model_name}")
    ModelClass = getattr(module, class_name)
    model = ModelClass(variables)
    conversation_with_summary = model.setup_model()

    print("Type 'exit' to stop\n---------------------------------------")

    while True:
        # Make sure the user inputs something
        text = ""
        while text == "":
            text = input("\nUser: ")
        # Exit the program if the user types 'exit'
        if text.lower() == "exit" or text.lower() == "quit" or text.lower() == "'exit'": 
            break
        model.run_model(text, conversation_with_summary)

    print("Goodbye!")


if __name__ == "__main__":
    main(sys.argv[1])
