from models.base_model import BaseModel
from icecream import ic
import ray



class Snorkel(BaseModel):
    def __init__(self, variables: dict[str, any]):
        super().__init__(variables)
        ray.init()

    def setup_model(self):
        """
        Setup the model.
        """
        # Setup template (which works best is different for each model)
        self.template = f"<s>[INST]{self.template}[/INST] Model answer</s>" + '\n\nCurrent conversation:\n{history}\nuser: {input}\nassistant:'

        conversation_with_summary = self.setup_conversation_chain(template=self.template)
    
        return conversation_with_summary
    
    @ray.remote
    def run_model(self, text:str, conversation_with_summary) -> str:
        """
        Run the model, print the response and return the response.
        """
        response = conversation_with_summary.predict(input=text)
        return response
