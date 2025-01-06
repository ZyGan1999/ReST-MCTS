import torch

#print(torch.__version__)
#print(torch.version.cuda)
#print(torch.cuda.is_available())

from MCTS.task import *
#from models.model import init_model




print(f'[1] {torch.cuda.is_available()}')

#init_model()

print(f'[2] {torch.cuda.is_available()}')


question = "Calculate the sum of the first 10 prime numbers."
task = MCTS_Task(question, 'llama', 'local', lang='en', iteration_limit=5)
output = task.run()
print(output[0]['solution'])

