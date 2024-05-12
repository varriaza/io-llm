python -m venv io-llm
source io-llm/bin/activate

export CMAKE_ARGS="-DLLAMA_CUBLAS=on" 
export FORCE_CMAKE=1 

pip install -r requirements.txt
clear