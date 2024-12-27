# LLMs_from_scratch
Learning records for building a large language model from scratch

- Book: [Build a Large Language Model (From Scratch)](https://www.manning.com/books/build-a-large-language-model-from-scratch?utm_source=raschka&utm_medium=affiliate&utm_campaign=book_raschka_build_12_12_23&a_aid=raschka&a_bid=4c2437a0&chan=mm_github)
- Github repo: [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)


I implemented the codes in the fantastic book. All the codes are run with a Macbook Pro (2020 version). Thus, I can use only two types of devices: `cpu` and `mps`. Perhaps, the Macbook is a little old that is equipped with M1 chip. The training speed of the`mps` mode is even slower than the `cpu` mode.

### Records

 - **Section 2**:

- [x] Understanding word embeddings

- [x] Tokenizing text

- [x] Converting tokens into token IDs

- [x] Adding special context tokens

- [x] Byte pair encoding

- [x] Data sampling with a sliding window

- [x] Creating token embeddings

- [x] Encoding word positions

 - **Section 3**:

- [x] Capturing data dependencies with attention mechanisms

- [x] Implementing self-attention with trainable weights

- [x] Hiding feature words with causal attention

- [x] Multi-head attention

 - **Section 4**:

- [x] Activations and layer normalization

- [x] Adding shortcut connections

- [x] Build a Transformer block

- [x] Coding the GPT model

- [x] Generating text

 - **Section 5**:

- [x] Evaluating generative text model

- [x] Training an LLM

![](ch05/train_plot.png)

- [x] Greedy search and Top-k sampling

- [x] Load pretrained weights from OpenAI

**PS**: When I download the source model data from OpenAI, the downloading procedure is always frequently broken. Therefore, I tried multiple times and finally collect both the `small` and `media` models. These models are uploaded to Baidu Cloud for your convenience.


|Model Size|OpenAI Sources|Converted (Pytorch version)|
|:-----:|:-----:|:-----:|
|`small`|[Baidu Cloud](https://pan.baidu.com/s/1BMpqgnkceMsNYGqOzNybxA?pwd=d3wu) (psw: d3wu)| [Baidu Cloud](https://pan.baidu.com/s/1_oL4DSRfWg6wBmSJ6vDISA?pwd=r4hq) (psw: r4hq)|
|`media`|[Baidu Cloud](https://pan.baidu.com/s/1Ih1A0UQPUsAOdwT0eoGmhw?pwd=qqqj) (psw: qqqj) | [Baidu Cloud](https://pan.baidu.com/s/1n_2WndBnEviIhO3X6MShCg?pwd=8whr) (psw: 8whr)|


 - **Section 6**:

- [x] Prepare spam email dataset and dataloader

- [x] Fine-tune the model on supervised data

- [x] Use the LLM as a spam classifier

 - **Section 7:**

- [x] Prepare a dataset for supervised instruction fine-tuning

- [x] Organize data into training batches

- [x] Finetune the LLM on instruction data

**PS**: It is challenging for me to train with `gpt2-media (355M)` model. Thefore, I still use the light-weight `gpt2-small (124M)`. So, it is no superise that the predictions of the finetuned model perform bad.

