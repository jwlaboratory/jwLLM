# jwLLM: a LLM inference server built in Cpp

Read the blog: https://jwlabs.vercel.app/post/jwllm-part1


### Usage
1. Download safetensors into /data from https://huggingface.co/openai-community/gpt2
2. Run `make`
3. Run `./build/bin/jwLLM`
4. Enjoy GPT-2!




### Ideas for exploration

- How much can we multithread? multi threading attention, the tokenizer, and the vector ops
for caching, any optimal order of accessign the stuff? maybe focusing on one region accross all threads first

- KV caching with MMAP

- continious batching

- paged kv cache

- scheduling algos for batches


# from class so far
- multithreading
- process scheduling
- mmap for file loading
