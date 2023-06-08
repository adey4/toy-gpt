# Toy-GPT
#### Toy-GPT is a decoder-only transformer built from scratch using PyTorch, trained to generate natural language similar to [input.txt](input.txt)
#
### Dependencies
- Python3
- PyTorch: `conda install pytorch torchvision -c pytorch`

### Limitations
- Training takes ~1 hr on Apple's M1 Pro Chip
- Language generation quality is limited by compute

### Example Output
Trained on Shakespearean text:
```
SICINIUS:
Is it strange?

Herald:
He's deceited, and children from his new spid
Then whomen he dares to him: were he worse.

BRUTUS:
You have pirtly not him.

MENENIUS:
What's the prisoner have not a silfa?

MONTAGUE:
O, and both shame, Menenius. Stanless, Thou art purpose;
And said thou pen for thy melting there,--

BENVOLIO:
Two sir, the earth proofs rids too come hither;
I thank you out, as thought sook for Ireland,

FRIAR LAURENCE:
His son, do your morself, that leaven your honours
Sufferable in more and suffer five.
A horse! High-graced York rights. And bother Montague
```

### Sources
- ["Attention is All You Need" by Vaswani et al.](https://doi.org/10.48550/arXiv.1706.03762)
- ["Let's build GPT: from scratch, in code, spelled out." by Andrej Karpathy](https://www.youtube.com/watch?v=kCc8FmEb1nY)
