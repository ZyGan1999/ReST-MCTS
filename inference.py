import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# transformers > 4.37.0
model_name = "/data/ganzeyu/Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=False)
# Positive
prompt = 'Answer the following questions.\n\nQuestion: John buys 3 puzzles.  The first puzzle has 1000 pieces.  The second and third puzzles have the same number of pieces and each has 50% more pieces.  How many total pieces are all the puzzles?\nAnswer: The second puzzle has 1000*.5=<<1000*.5=500>>500 more pieces than the first\nSo it has 1000+500=<<1000+500=1500>>1500 total pieces\nThat means those two puzzles have 1500*2=<<1500*2=3000>>3000 pieces\nSo in total there were 3000+1000=<<3000+1000=4000>>4000 pieces So the answer is 4000.\n\nQuestion: Mrs. Young buys a 300 piece puzzle set for her three sons.  She divides the pieces evenly to the three boys.  Reyn can place 25 pieces into the puzzle picture.  Rhys places twice as much as Reyn.  Rory places three times as much as Reyn. Altogether, how many puzzle pieces are still left to be placed?\nAnswer: Each son receives 300 / 3 = <<300/3=100>>100 pieces.\nRhys places 25 x 2 = <<25*2=50>>50 pieces.\nRory places 25 x 3 = <<25*3=75>>75 pieces.\nAll three boys place 25 + 50 + 75 = <<25+50+75=150>>150 pieces.\nThe boys have not placed 300 - 150 = <<300-150=150>>150 pieces. So the answer is 150.\n\nQuestion: James buys 2 puzzles that are 2000 pieces each.  He anticipates for these larger puzzles he can do 100 pieces every 10 minutes.  So how long would it take to finish both puzzles?\nAnswer: His speed means he can do 10/100=<<10/100=.1>>.1 minute per piece\nSo he can finish 1 puzzle in 2000*.1=<<2000*.1=200>>200 minutes\nSo these two puzzles take 200*2=<<200*2=400>>400 minutes So the answer is 400.\n\nQuestion: Pablo likes to put together jigsaw puzzles. He can put together an average of 100 pieces per hour. He has eight puzzles with 300 pieces each and five puzzles with 500 pieces each. If Pablo only works on puzzles for a maximum of 7 hours each day, how many days will it take him to complete all of his puzzles?\nAnswer: First find how many pieces are total in each puzzle. 8 puzzles * 300 pieces each = <<8*300=2400>>2400 pieces.\nNext, 5 puzzles * 500 pieces each = <<5*500=2500>>2500 pieces.\nAll of the puzzles have 2400 pieces + 2500 pieces = <<2400+2500=4900>>4900 pieces total.\nHe will work a maximum of 7 hours each day * 100 pieces per hour = <<7*100=700>>700 pieces per day.\nSo he will end up taking 4900 pieces total / 700 pieces per day = <<4900/700=7>>7 days total. So the answer is 7.\n\nQuestion: Kalinda is working on a 360 piece puzzle with her mom. Kalinda can normally add 4 pieces per minute. Her mom can typically place half as many pieces per minute as Kalinda.  How many hours will it take them to complete this puzzle?\nAnswer:'
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )
text = prompt
print("***input***:",text)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("***response***:",response)
