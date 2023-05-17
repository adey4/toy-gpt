# Toy-GPT
#### Toy-GPT is a decoder-only transformer built from scratch using PyTorch, trained to generate natural language similar to input.txt
#
### Dependencies
- Python3
- Pytorch: `conda install pytorch torchvision -c pytorch`

### Limitations
- Expensive to train: ~1 hr on M1 Mac
- Training efficiency can be improved with further optimizations

### Example Output
Trained on Shakespearean text:
```
LUCIO:
We muse hath resistes him so sovere: son't his other wrough stands of coverent sh'd: he has here, and stand it
and poor exceeder or a Henry's last, stay
not in faith, forewell's base of graves, thanks, happy comparel,
warment fully: may as face by the courst, that strangth
errise hath breathed. Mastings come to Valenting.

HERMIONE:
Well have been bolly poor late
Is the lords.

ABELLA:
Let's found: I will kind him;
I do braw'sy him business wherein far his face.

LUCENTIO:
He is last afford: make him diseably to London, 
Take him great Hastings, boldness in his natic keeps,
To oftragn lost me ready glust through the house.
Why chose that I dares it be a Montague.

MONTAGUE:
Woe's Claudly Haste of his own at last the Volscient,
And seen'd helpit: bearn to do it be, and most hop,
Miscause's more conterar than without this lambs
Shall down appla fortune flight flowers.

FRIAR LAUAURENCE:
His son, do your morself, that leaven your honours
Sufferable in more and suffer five.
A horse! High-graced York rights. And bother Montague
```

### Sources
- ["Attention is All You Need" by Vaswani et al.](https://doi.org/10.48550/arXiv.1706.03762)
- [Andrej Karpathy on Building GPT from Scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)
