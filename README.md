# jwLLM: a LLM inference server built in Cpp


We are building engine for
https://huggingface.co/openai-community/gpt2

External packages are not committed as header files in this repository; the Makefile downloads them into the build directory during setup. Include external headers normally in the source code, and add the corresponding package to the Makefile.

### Roadmap

**Goal 1: Working inference engine that can run GPT-2**
Target: Week 1 / Aug 24-Aug 29
In class: Intro to OS + syllabus + history

#### Tokenization

- [ ] Implement GPT-2 byte-level text preprocessing
- [ ] Implement BPE tokenization using the merge-rank table
- [ ] Encode text into GPT-2 token IDs
- [ ] Decode token IDs back into text, including byte-level characters
- [ ] Handle unknown, end-of-text, empty, and non-ASCII input cases
- [ ] Add tokenizer tests with known input/output examples

#### Tensor and numerical operations

- [ ] Define a reusable tensor type with shape, storage, indexing, and ownership rules
- [ ] Add the operations needed by GPT-2: matrix multiplication, addition, scaling, reshape, transpose, and softmax
- [ ] Add layer normalization and GELU
- [ ] Validate shapes and report useful errors
- [ ] Add unit tests for each operation, including numerical tolerance checks

#### GPT-2 transformer

- [ ] Add token and positional embeddings
- [ ] Implement masked multi-head self-attention with a causal mask
- [ ] Implement the GPT-2 feed-forward network
- [ ] Implement residual connections and pre-layer normalization
- [ ] Build a configurable stack of transformer blocks
- [ ] Project the final hidden state to vocabulary logits
- [ ] Compare intermediate outputs and logits with a reference implementation

#### Text generation

- [ ] Implement autoregressive next-token inference
- [ ] Respect the model context length
- [ ] Stop at the end-of-text token or a configured maximum number of tokens
- [ ] Support greedy decoding first
- [ ] Add temperature and top-k sampling after greedy decoding is verified
- [ ] Add a reproducible random seed for sampling tests




### Build instructions
build the app
```
make
```

run
```
./build/my_app
```

run tests
```
make test
```

