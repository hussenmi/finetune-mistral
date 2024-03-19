# Fine-Tune Mistral-7B-Instruct-v0.2-GPTQ Using QLoRA

In this repository, I fine-tune the an open source LLM (specifically, the  Mistral-7B-Instruct-v0.2-GPTQ) on comments and replies gathered from Reddit using QLoRA.

I have uploaded the data and fine-tuned model on the Hugging Face Hub.

The data can be found [here](https://huggingface.co/datasets/hussenmi/reddit_comments).

The original base model can be found [here](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GPTQ).

The fine-tuned model can be found [here](https://huggingface.co/hussenmi/fungpt-ft).

## Description of the Fine-Tuning Approach

There are a couple of different methods to fine-tune a Large Language Model (LLM). One common method is \textbf{full fine-tuning}. The process results in a new version of the model with updated weights. One caveat with this process is that full fine-tuning requires enough memory and computing power to process all the gradients and other components being updated during training.

In order to work against this constraint, we have another method called \textbf{parameter-efficient fine-tuning}. In this method, we only update a small set of parameters, which saves us a lot of computational power and memory. One method of doing this is called LoRA **(Low-Rank Adaptation)**.

### LoRA

LoRA (Low-Rank Adaptation) is a method used for parameter-efficient fine-tuning of large language models. It is designed to update only a small set of parameters, reducing the computational power and memory requirements compared to full fine-tuning.

The idea behind LoRA is to identify a subset of parameters in the model that can be updated to adapt the model to a specific task or domain. This subset of parameters is referred to as the ``target modules.'' By updating only these target modules, LoRA achieves parameter-efficient fine-tuning.

The key concept in LoRA is the low-rank approximation of the weight matrices in the target modules. Instead of updating the full weight matrices, LoRA decomposes them into low-rank factors. This decomposition reduces the number of parameters that need to be updated, resulting in significant memory and computational savings.

During the fine-tuning process, LoRA updates the low-rank factors of the target modules using gradient descent. The gradients are computed using backpropagation through the model, similar to traditional fine-tuning methods. However, since LoRA only updates a small set of parameters, the computational cost is significantly reduced.

Let's discuss this using an example. In transformers, there are three vectors that are generated in each head: $Q, K$, and $V$. In order to generate these vectors, we have matrices associated with each of them: $W^Q, W^K$, and $W^V$. Since in LLMs, we have multiple attention heads, we'll also have more of these matrices as well. To illustrate what LoRA does, let's just look at one of the matrices, $W^Q$.

For our discussion, let's assume this weight matrix is a $7$ x $7$ matrix. We have 49 elements. This is an example of what the matrix ($W^Q$) could look like:


$$\begin{pmatrix}
4 & 7 & 2 & 9 & 1 & 5 & 3 \\
6 & 3 & 8 & 2 & 7 & 4 & 1 \\
5 & 9 & 1 & 3 & 8 & 6 & 2 \\
7 & 2 & 4 & 6 & 9 & 1 & 5 \\
3 & 8 & 6 & 1 & 4 & 2 & 7 \\
2 & 5 & 9 & 7 & 3 & 8 & 6 \\
1 & 6 & 3 & 4 & 5 & 7 & 9 \\
\end{pmatrix}$$


If we want to use full fine-tuning, we'd have to update all of these elements. And we can assume as the matrix gets bigger, we need to update more values as well. One idea to reduce the number of elements to update is, instead of having one big matrix, why don't we have two smaller matrices, when multiplied will give us the same dimension as the original matrix. Let's call these two matrices $A$ and $B$, and in our scenario, their dimensions will be $7$ x $r$ and $r$ x $7$ simultaneously. These matrices are lower in rank when compared to the original matrix. They have a rank of $r$. So they could, for example, be a $7$ x $2$ and a $2$ x $7$ matrix. This is what $A$ and $B$ could look like:

$$\begin{pmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22} \\
a_{31} & a_{32} \\
a_{41} & a_{42} \\
a_{51} & a_{52} \\
a_{61} & a_{62} \\
a_{71} & a_{72} \\
\end{pmatrix}$$

$$\begin{pmatrix}
b_{11} & b_{12} & b_{13} & b_{14} & b_{15} & b_{16} & b_{17} \\
b_{21} & b_{22} & b_{23} & b_{24} & b_{25} & b_{26} & b_{27} \\
\end{pmatrix}$$


When we multiply them, we get a $7$ x $7$ matrix. This matrix is called the adapter. It could be element-wise added to the original weight matrix, $W^Q$, and the result will be the new fine-tuned version of the weight matrix that we can use in our model. Apparently, this would work as a fine-tuning method because the values in these low-rank matrices, $A$ and $B$ are learned during the fine-tuning process, so they have the new information from the new data. When we add this new matrix to the original matrix, we have both the pre-trained information and the new information extracted from the fine-tuning data.

This method is great and saves us a lot of computational work and allows us to fine-tune our model efficiently. Lucky for us, researchers have also come up with a way to make this even more efficient by adding quantization to the mix to create **QLoRA (Quantized Low-Rank Adaptation)**.

Quantization is a process that reduces the precision of numerical values to save memory and computational resources. This could involve rounding, clustering values, and mapping to a set of representable values while preserving as much of the original information as possible.
