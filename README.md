# jwLLM: a LLM inference server built in Cpp


We are building engine for
https://huggingface.co/openai-community/gpt2

each week of work is aimed to align with learning objectives from Operating Systems 439


plans: (mappign OS concepts to project)
-> vLLM --- > virtual memory
-> file system -- >flexgen, efficient SSD / kv transfer etc
-> concurrency: multi threading 
-> CPU scheduling : batch scheduling 


External packages are not committed as header files in this repository; the Makefile downloads them into the build directory during setup. Include external headers normally in the source code, and add the corresponding package to the Makefile.



### tokenizer
improvements:
-> mmap
-> processes, fork, etc
-> simd



