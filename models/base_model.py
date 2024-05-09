from icecream import ic
from huggingface_hub import hf_hub_download
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory, ConversationBufferMemory, ConversationKGMemory
from langchain.chains import ConversationChain
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler
)


class BaseModel:
    def __init__(self, variables: dict[str, any]):
        self.variables = variables
        # Load variables and standardize to lowercase
        self.ai_name = self.variables["name"]
        self.model_name = self.variables["model"].lower()
        self.quality = self.variables["quality"].lower()
        self.model_filename = self.variables[self.quality]["model_filename"]
        self.model_path = "./model_files/" + self.model_filename
        self.memory_type = self.variables["memory_type"]
        self.mode = self.variables["mode"].lower()
        if self.mode == "creative":
            self.temperature = .25
        elif self.mode == "balanced":
            self.temperature = .5
        elif self.mode == "factual":
            self.temperature = .75
        else:
            raise ValueError(f"Mode '{self.mode}' not recognized. It must be 'creative', 'balanced', or 'factual'.")
        
        # The context window size. This is history + input + response
        self.context_window_size = 1024*(int(self.variables["context_size"]))
        # self.context_window_size = 1024*100
        # The maximum number of tokens to store in memory
        self.max_memory_token_limit = self.context_window_size/2
        # The maximum number of tokens to generate per response
        self.max_tokens_per_response = self.context_window_size/2
        # The number of batches to run in parallel
        self.n_batch = 512

        # Set how the AI will answer the user
        self.template = self.variables["style"]

        # Set if the AI will stream its answers
        self.use_streaming = self.variables["use_streaming"]
        # check if use_streaming is a string
        if isinstance(self.use_streaming, str): 
            # Turn the string into a boolean
            if self.use_streaming.lower() == "true":
                self.use_streaming = True
            elif self.use_streaming.lower() == "false":
                self.use_streaming = False
            else:
                raise ValueError(f"Use streaming '{self.use_streaming}' not recognized. It must be 'true' or 'false'.")
        

    def is_model_missing(self, model_filename: str) -> bool:
        """
        Test if the model is downloaded.
        Args:
            model_filename: The name of the model file.
        Returns:
            is_missing: A boolean indicating if the model is missing.
        """
        is_missing = True
        # Check if a file with the model name exists in the model_files directory
        try:
            with open("./model_files/" + model_filename, "r") as _:
                is_missing = False
                print(f"Model {model_filename} already downloaded.")
        except FileNotFoundError:
            pass
        return is_missing

    def download_model(self, repo_id:str, model_filename: str):
        """
        Download the model from the Hugging Face Hub.
        Args:
            model_filename: The name of the model file.
        """
        # Download the model from the Hugging Face Hub
        hf_hub_download(repo_id=repo_id, filename=model_filename, local_dir="./model_files/", local_dir_use_symlinks=False)


    def setup_conversation_chain(self, template: str, debug: bool = False) -> ConversationChain:
        """
        Setup the conversation chain.
        """
        # Download the model if it is missing
        if self.is_model_missing(self.model_filename):
            self.download_model(self.variables["repo_id"], self.model_filename)

        if self.use_streaming:
            # Setup streaming of answers
            callbackManager = CallbackManager([StreamingStdOutCallbackHandler()])
            # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
            llm = LlamaCpp(
                model_path=self.model_path,  # Download the model file first
                n_gpu_layers=-1,         # The number of layers to offload to GPU, if you have GPU acceleration available
                verbose=debug, # Only set to True if you want to see the output to debug
                max_tokens=self.max_tokens_per_response, # Max tokens used per response. This will cut off the response if it exceeds the token limit.
                n_batch=self.n_batch, # The number of batches to run in parallel.
                n_ctx=self.context_window_size, # The context window size. This is history + input + response
                temperature=self.temperature, # The temperature of the model. 0.0 is whimsical, 1.0 is factual
                callback_manager=callbackManager, # CAN BREAK RESPONSES: Steaming works but after the first response, the model starts repeating past questions.
            )
        else:
            # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
            llm = LlamaCpp(
                model_path=self.model_path,  # Download the model file first
                n_gpu_layers=-1,         # The number of layers to offload to GPU, if you have GPU acceleration available
                verbose=debug, # Only set to True if you want to see the output to debug
                max_tokens=self.max_tokens_per_response, # Max tokens used per response. This will cut off the response if it exceeds the token limit.
                n_batch=self.n_batch, # The number of batches to run in parallel.
                n_ctx=self.context_window_size, # The context window size. This is history + input + response
                temperature=self.temperature, # The temperature of the model. 0.0 is whimsical, 1.0 is factual
            )


        if self.memory_type.lower() == "standard":
            # Setting k=2, will only keep the last 2 interactions in memory
            # Setting max_token_limit=2048, will only keep the last 2048 tokens in memory
            memory = ConversationSummaryBufferMemory(k=self.variables["memory_length"], llm=llm, max_token_limit=self.max_memory_token_limit) 
        elif self.memory_type.lower() == "knowledge graph":
            memory = ConversationKGMemory(llm=llm)
        elif self.memory_type.lower() == "everything":
            memory = ConversationBufferMemory(k=self.variables["memory_length"], llm=llm, max_token_limit=self.max_memory_token_limit) 
        else:
            raise ValueError(f"Memory type '{self.memory_type}' not recognized.")
        
        # Setup prompt
        prompt = PromptTemplate(input_variables=['history', 'input'], template=self.template)

        conversation_with_summary = ConversationChain(
            llm=llm, 
            memory=memory,
            prompt=prompt,
            verbose=debug
        )

        return conversation_with_summary
