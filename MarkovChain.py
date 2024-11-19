import MarkovChainImproved

with open("C:\\Users\\jafar\\cyberflicker\\resources\\independence.txt") as f:
    text = f.read()

text_model = markovify.Text(text)
for i in range(5):
    print(text_model.make_sentence())
print("Short sentence")
for i in range(3):
    print(text_model.make_short_sentence(140))

