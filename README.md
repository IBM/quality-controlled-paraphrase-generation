# Quality Controlled Paraphrase Generation (ACL 2022)
> Paraphrase generation has been widely used in various downstream tasks. Most tasks benefit mainly from high quality paraphrases, namely those that are semantically similar to, yet linguistically diverse from, the original sentence. Generating high-quality paraphrases is challenging as it becomes increasingly hard to preserve meaning as linguistic diversity increases. Recent works achieve nice results by controlling specific aspects of the paraphrase, such as its syntactic tree. However, they do not allow to directly control the quality of the generated paraphrase, and suffer from low flexibility and scalability. 

<img src="/assets/images/ilus.jpg" width="40%"> 

> Here we propose `QCPG`, a quality-guided controlled paraphrase generation model, that allows directly controlling the quality dimensions. Furthermore, we suggest a method that given a sentence, identifies points in the quality control space that are expected to yield optimal generated paraphrases. We show that our method is able to generate paraphrases which maintain the original meaning while achieving higher diversity than the uncontrolled baseline.

## Training Evaluation and Inference
The code for training evaluation and inference for both `QCPG` and `QP` models is located in the dedicated directory. scripts for reproducing the experiments can be found in the scripts directories inside the models directory. 

<img src="/assets/images/arch.png" width="90%"> 

## Trained Models

```
!!! Notice !!! Our results show that on avarage QCPG is follwing the quality conditions and capable of generating higher quality greedy-sampled paraphrases then finetuned model. It does not mean it will output perfect paraphrases all the time!!! In practice, for best preformence, we highly reccomend: (1) Find the right quality control values (2) Use more sophisticated sampling methods (3) Apply post-generation monitoring and filtering. 
```


[`qcpg-questions`](https://huggingface.co/ibm/qcpg-questions)

[`qcpg-sentences`](https://huggingface.co/ibm/qcpg-sentences)

[`qcpg-captions`](https://huggingface.co/ibm/qcpg-captions)

## Citation
```
@inproceedings{bandel-etal-2022-quality,
    title = "Quality Controlled Paraphrase Generation",
    author = "Bandel, Elron  and
      Aharonov, Ranit  and
      Shmueli-Scheuer, Michal  and
      Shnayderman, Ilya  and
      Slonim, Noam  and
      Ein-Dor, Liat",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.45",
    pages = "596--609",
    abstract = "Paraphrase generation has been widely used in various downstream tasks. Most tasks benefit mainly from high quality paraphrases, namely those that are semantically similar to, yet linguistically diverse from, the original sentence. Generating high-quality paraphrases is challenging as it becomes increasingly hard to preserve meaning as linguistic diversity increases. Recent works achieve nice results by controlling specific aspects of the paraphrase, such as its syntactic tree. However, they do not allow to directly control the quality of the generated paraphrase, and suffer from low flexibility and scalability. Here we propose QCPG, a quality-guided controlled paraphrase generation model, that allows directly controlling the quality dimensions. Furthermore, we suggest a method that given a sentence, identifies points in the quality control space that are expected to yield optimal generated paraphrases. We show that our method is able to generate paraphrases which maintain the original meaning while achieving higher diversity than the uncontrolled baseline. The models, the code, and the data can be found in https://github.com/IBM/quality-controlled-paraphrase-generation.",
}
```
