# README 

In this module, we explore the method of Inference-Time Compute Scaling. This method allows us to provide reasoning capabilities to the large language model without tweaking the base model, but by increasing the computational resources spent during inference. We will explore 2 prompting techniques in this lecture:

(1) Few-shot Chain of Thought Reasoning
Reference Paper: https://arxiv.org/abs/2201.11903
(2) Zero-shot Chain of Thought Reasoning
Reference Paper: https://arxiv.org/abs/2205.11916


We will cover both the broader learnings from both the papers as well as the nuances which only become clear upon reading the paper closely. Particularly, we will see how reasoning can be considered as an emergent property of the model size. You will also become familiar with arithmetic, logical and symbolic reasoning datasets in this lecture.


Here are the Google Colab Notebooks which have been referenced in the video:

Hands-on Project 1: Exploring the effect of model size on the accuracy of chain of thought prompting

Hands-on Project 2: Explore the difference in performance of 3 methods: (1) Few shot prompting (baseline), (2) Few shot chain of thought prompting, (3) Zero shot chain of thought prompting


Hands-on Project 3: Applying Zero-Shot Chain of Thought Prompting for an LLM built from scratch
Inspiration: https://huggingface.co/rasbt/llama-3....
