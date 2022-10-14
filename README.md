# Overview

This will be the repository for the software supporting the paper [*Cross-Dialect Social Media Dependency Parsing for Social Scientific Entity Attribute Analysis*](https://aclanthology.org/2022.wnut-1.4/) by [Chloe Eggleston](https://chloes.computer/) ([@nu11us](https://github.com/nu11us)) and [Brendan O'Connor](http://brenocon.com/) ([@brendano](https://github.com/brendano)), Proceedings of the Workshop on Noisy User-generated Text (W-NUT) at COLING 2022.


A copy of the paper is included here; see also the [ACL Anthology](https://aclanthology.org/2022.wnut-1.4/). Bibtex:

```
@inproceedings{eggleston-oconnor-2022-cross,
    title = "Cross-Dialect Social Media Dependency Parsing for Social Scientific Entity Attribute Analysis",
    author = "Eggleston, Chloe  and
      O{'}Connor, Brendan",
    booktitle = "Proceedings of the Eighth Workshop on Noisy User-generated Text (W-NUT 2022)",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.wnut-1.4",
    pages = "38--50",
    abstract = "In this paper, we utilize recent advancements in social media natural language processing to obtain state-of-the-art syntactic dependency parsing results for social media English. We observe performance gains of 3.4 UAS and 4.0 LAS against the previous state-of-the-art as well as less disparity between African-American and Mainstream American English dialects. We demonstrate the computational social scientific utility of this parser for the task of socially embedded entity attribute analysis: for a specified entity, derive its semantic relationships from parses{'} rich syntax, and accumulate and compare them across social variables. We conduct a case study on politicized views of U.S. official Anthony Fauci during the COVID-19 pandemic.",
}
```

# Paper errata

Paragraph 3 of Section 3.4 discusses the construction of the Tweebank v2 AAE and MAE splits of the test set.  It should be changed to the following:

> In order to measure disparity on the fine-tuning source, we measure the
> relative error of both the TwitterAAE dependencies and use the Twitter-AAE
> demographic dialect inference model to extract dialect-specific subsets of
> the Tweebank v2 test set based on whether the largest demographic proportion
> was MAE or AAE, yielding 672 and 163 tweets respectively (the remaining 366
> tweets belong to neither subset).  We also analyze the TwitterAAE
> dependencies in the same way, which provides 250 tweets of both MAE and AAE
> respectively.

All per-dialect accuracy results and disparities reported in the paper were evaluated on the above sets of messages.

The published text for this paragraph describes a different splitting method to partition the test set according to whether the AAE or MAE proportion is higher; this criterion results in a supersets of the above selections, since the demographic model has four classes.
