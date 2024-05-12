from models.base_model import BaseModel
from icecream import ic


class Snorkel(BaseModel):
    def __init__(self, variables: dict[str, any]):
        super().__init__(variables)

    def setup_model(self):
        """
        Setup the model.
        """
        # Setup template (which works best is different for each model)
        self.template = f"<s>[INST]{self.template}[/INST] Model answer</s>" + '\n\nCurrent conversation:\n{history}\nuser: {input}\nassistant:'

        conversation_with_summary = self.setup_conversation_chain(template=self.template)
    
        return conversation_with_summary
    

