## GPT Text Vis:
A visualizer in Python for GPT-2's processing of input text

I think language models' ability to generate text is less interesting than the statistical knowledge of language use that they contain, so I made this visualizer to more easily understand what's going on under the hood as GPT-2 processes an input text sequence. Hopefully it's pretty easy to use—just copy or type your text into the designated box, press the button, then hover over each token in the processed text to see the surprisal of that token (-log(p(token)) and the entropy of the next prediction. The color of the processed tokens correspond to their relative surprisal, with more red tokens being more surprising.

This visualizer is built with Python and PyQt6 and requires PyTorch and Transformers.

![example](/example.jpeg "Example Prediction")
