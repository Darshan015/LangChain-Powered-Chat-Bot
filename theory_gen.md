Gen AI: Technology which basically predict and generate the output as text, image, audio, video by using the large datasets
ML: Based on human/user input it predicts and recommend and in case of ML user have to explicitely define the function
ex: Movie recommendation system on Netflix
DL: The input from the user goes through several hidden layer(Neural network) to understand and generate the output, here Neural network learn by itself
ex: Face recognition
LLM(Large Language Model): Generate the text
LIM(Large Image model)
LVM(Large Vision model)
LAM(large audio model)
################################################################
LLM:
Model/Artificial Intelligence which is pretrained with large set of text data.
Ex: Chatgpt, Claude, Grook etc..
Chatgpt: Chat-->Generative Pre-trained Transformer
Basically the model generates the next word from pre trained data using the transformer architecture
And the next best word will be predicted based on the highest probablity of the word from set of word pool

Tokenization:
basically LLM doesn't understand the text
Once the prompt is given, the sentence gets chopped and turn it to set of chunks or you can say list of string
A unnique number would be assigned to each chunck/string(Every LLM has their own dictionary where the token id would mapped to word)-->Token id would be received by the LLM and predict the output and send it back to user
ex:
import tiktoken
text = "My name is Darshan"
tokenizer = tiktoken.get_encoding('cl100k_base')
tokens = tokenizer.encode(text)
print(tokens)
res = tokenizer.decode(tokens)
print(res)

Context Window: How much data we can send to LLM and how much data LLM would send in response(or we would get from LLM)
Max token:all the model(open ai/anthropic) has max token assigned and you can't access token on top of that( also mentioned in pricing information)includes input/cache/output token
